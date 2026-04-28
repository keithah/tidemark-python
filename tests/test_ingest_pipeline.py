from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.ingest.pipeline import (
    IngestPipelineResult,
    TranscriptFixtureError,
    ingest_source_to_db,
    load_fixture_transcript,
)
from tidemark.ingest.segments import SegmentIngestError
from tidemark.search import TranscriptDatabaseEmpty, search_transcript_db
from tidemark.transcribe import DeterministicTranscriber


def _make_tiny_wav(path: Path) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.20:sample_rate=8000",
            "-ac",
            "1",
            "-y",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return path


def _write_manifest(path: Path, segment_name: str, *, include_cue: bool = True) -> Path:
    lines = [
        "#EXTM3U",
        "#EXT-X-MEDIA-SEQUENCE:37",
    ]
    if include_cue:
        lines.append("#EXT-X-CUE-OUT:DURATION=15.0")
    lines.extend(
        [
            "#EXTINF:0.20,",
            segment_name,
            "#EXT-X-CUE-IN",
            "#EXT-X-ENDLIST",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fetch_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_ingest_source_to_db_persists_segments_words_markers_and_searches(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber(
            [
                ("hello", 0.03, 0.06, 0.9),
                ("tidemark", 0.07, 0.11, 0.8),
                ("search", 0.12, 0.16, None),
            ],
            language="en",
            engine="deterministic-fixture",
        ),
        source_url="fixture://integration/source.m3u8?token=secret",
    )

    assert isinstance(result, IngestPipelineResult)
    assert result.issues == ()
    assert len(result.segment_ids) == 1
    assert len(result.transcript_word_ids) == 3
    assert len(result.ad_event_ids) >= 1

    with sqlite3.connect(db_path) as conn:
        segment_rows = conn.execute(
            "SELECT id, sequence, start_ts, duration_seconds FROM segments ORDER BY id"
        ).fetchall()
        word_rows = conn.execute(
            "SELECT id, segment_id, segment_sequence, word_index, word_text, start_ts, end_ts, confidence "
            "FROM transcript_words ORDER BY id"
        ).fetchall()
        ad_rows = conn.execute(
            "SELECT id, marker_type, classification, source, tag, segment_seq, ts FROM ad_events ORDER BY id"
        ).fetchall()

    assert segment_rows == [(result.segment_ids[0], 37, 0.0, 0.2)]
    assert [row[0] for row in word_rows] == list(result.transcript_word_ids)
    assert [(row[1], row[2], row[3], row[4]) for row in word_rows] == [
        (result.segment_ids[0], 37, 0, "hello"),
        (result.segment_ids[0], 37, 1, "tidemark"),
        (result.segment_ids[0], 37, 2, "search"),
    ]
    assert [(row[5], row[6], row[7]) for row in word_rows] == [
        (pytest.approx(0.03), pytest.approx(0.06), 0.9),
        (pytest.approx(0.07), pytest.approx(0.11), 0.8),
        (pytest.approx(0.12), pytest.approx(0.16), None),
    ]
    assert any(row[2] == "AD_START" and row[4] == "#EXT-X-CUE-OUT" and row[5] == 37 for row in ad_rows)

    search_results = search_transcript_db(db_path, "tidemark search", context_seconds=1)
    assert len(search_results) == 1
    search_result = search_results[0]
    assert search_result.hit_start_ts == pytest.approx(0.07)
    assert search_result.hit_end_ts == pytest.approx(0.16)
    assert search_result.context_text == "hello tidemark search"
    assert search_result.word_ids == result.transcript_word_ids[1:]


def test_default_ingest_does_not_create_fingerprint_or_retained_audio_side_effects(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
    )

    assert result.issues == ()
    assert len(result.segment_ids) == 1
    assert len(result.transcript_word_ids) == 1
    assert not (tmp_path / "tidemark-audio").exists()
    assert _fetch_count(db_path, "songs") == 0
    assert _fetch_count(db_path, "fingerprint_cache") == 0
    assert _fetch_count(db_path, "retained_audio") == 0


def test_load_fixture_transcript_accepts_json_array(tmp_path: Path) -> None:
    fixture_path = tmp_path / "transcript.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"text": "hello", "start_offset": 0.0, "end_offset": 0.1, "confidence": 0.75},
                {"text": "tidemark", "start_offset": 0.2, "end_offset": 0.3},
            ]
        ),
        encoding="utf-8",
    )

    assert load_fixture_transcript(fixture_path) == (
        ("hello", 0.0, 0.1, 0.75),
        ("tidemark", 0.2, 0.3, None),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "json"),
        ({"text": "hello"}, "array"),
        ([{"text": "", "start_offset": 0.0, "end_offset": 0.1}], "text"),
        ([{"text": "hello", "start_offset": -0.1, "end_offset": 0.1}], "start_offset"),
        ([{"text": "hello", "start_offset": 0.2, "end_offset": 0.1}], "end_offset"),
        ([{"text": "hello", "start_offset": 0.0, "end_offset": 0.1, "confidence": 1.1}], "confidence"),
    ],
)
def test_load_fixture_transcript_rejects_malformed_field_only(tmp_path: Path, payload: object, message: str) -> None:
    fixture_path = tmp_path / "transcript.json"
    if isinstance(payload, str):
        fixture_path.write_text(payload, encoding="utf-8")
    else:
        fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TranscriptFixtureError, match=message) as exc_info:
        load_fixture_transcript(fixture_path)

    error_message = str(exc_info.value)
    assert "hello" not in error_message
    assert str(fixture_path) not in error_message


def test_network_source_url_is_rejected_before_db_work(tmp_path: Path) -> None:
    db_path = tmp_path / "tidemark.sqlite3"

    with pytest.raises(SegmentIngestError, match="network URL"):
        ingest_source_to_db(
            "https://example.test/private.m3u8?token=secret",
            db_path=db_path,
            transcriber=DeterministicTranscriber([]),
        )

    assert not db_path.exists()


def test_invalid_media_preserves_segment_row_and_returns_redacted_decode_issue(tmp_path: Path) -> None:
    bad_media = tmp_path / "private-bad-segment.wav"
    bad_media.write_bytes(b"not a wav segment")
    manifest = _write_manifest(tmp_path / "private-playlist.m3u8", bad_media.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("private phrase", 0.0, 0.1, None)]),
        source_url="fixture://integration/private.m3u8?token=secret",
    )

    assert len(result.segment_ids) == 1
    assert result.transcript_word_ids == ()
    assert len(result.ad_event_ids) >= 1
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.phase == "decode"
    assert issue.segment_sequence == 37
    assert "ffmpeg" in issue.message
    assert "private phrase" not in issue.message
    assert "token=secret" not in issue.message
    assert "private" not in issue.message
    assert _fetch_count(db_path, "segments") == 1
    assert _fetch_count(db_path, "transcript_words") == 0


def test_include_manifest_markers_false_writes_no_ad_events(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
    )

    assert result.ad_event_ids == ()
    assert _fetch_count(db_path, "ad_events") == 0


def test_no_match_search_returns_empty_tuple(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
    )

    assert search_transcript_db(db_path, "not present") == ()


def test_no_match_search_on_decode_only_db_has_no_transcripts(tmp_path: Path) -> None:
    bad_media = tmp_path / "bad-segment.wav"
    bad_media.write_bytes(b"not media")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", bad_media.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([]),
        include_manifest_markers=False,
    )

    with pytest.raises(TranscriptDatabaseEmpty, match="transcript_words"):
        search_transcript_db(db_path, "anything")
