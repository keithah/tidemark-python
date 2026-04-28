from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.audio import decode_segment_audio
from tidemark.ingest import resolve_segments
from tidemark.store import insert_segment, insert_transcript_words, get_transcript_words_for_segment, migrate
from tidemark.transcribe import DeterministicTranscriber, WordToken


def _make_tiny_media_segment(path: Path) -> Path:
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


def _write_one_segment_manifest(manifest: Path, segment_name: str) -> Path:
    manifest.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-MEDIA-SEQUENCE:37",
                "#EXTINF:0.20,",
                segment_name,
                "#EXT-X-ENDLIST",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def _persist_segment(conn: sqlite3.Connection, segment) -> int:
    return insert_segment(
        conn,
        source_url=segment.source_url,
        sequence=segment.sequence,
        resolved_uri=segment.resolved_uri,
        local_path=segment.local_path,
        start_ts=segment.start_ts,
        duration_seconds=segment.duration_seconds if segment.duration_seconds is not None else 0.0,
        byte_length=segment.byte_length,
        sha256=segment.sha256,
        metadata=segment.metadata,
    )


def test_fixture_audio_transcribes_into_ordered_stored_word_timestamps(tmp_path: Path) -> None:
    media_path = _make_tiny_media_segment(tmp_path / "segment37.wav")
    manifest = _write_one_segment_manifest(tmp_path / "playlist.m3u8", media_path.name)
    source_url = "fixture://integration/source.m3u8?token=secret"
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    segment = resolve_segments(manifest, source_url=source_url)[0]
    segment_id = _persist_segment(conn, segment)
    chunk = decode_segment_audio(segment)
    result = DeterministicTranscriber(
        [
            ("hello", 0.03, 0.07, 0.92),
            ("tidemark", 0.11, 0.16, None),
        ],
        language="en",
        engine="deterministic-fixture",
    ).transcribe(chunk)

    row_ids = insert_transcript_words(
        conn,
        segment_id=segment_id,
        source_url=chunk.source_url,
        segment_sequence=chunk.segment_sequence,
        words=result.words,
    )
    records = get_transcript_words_for_segment(conn, segment_id)

    assert len(row_ids) == 2
    assert tuple(record.id for record in records) == row_ids
    assert [(record.segment_id, record.source_url, record.segment_sequence) for record in records] == [
        (segment_id, source_url, 37),
        (segment_id, source_url, 37),
    ]
    assert [(record.word_index, record.word_text) for record in records] == [(0, "hello"), (1, "tidemark")]
    assert [(record.start_ts, record.end_ts, record.confidence) for record in records] == [
        (pytest.approx(chunk.start_ts + 0.03), pytest.approx(chunk.start_ts + 0.07), 0.92),
        (pytest.approx(chunk.start_ts + 0.11), pytest.approx(chunk.start_ts + 0.16), None),
    ]


def test_invalid_deterministic_word_rejection_redacts_phrase_and_source_token(tmp_path: Path) -> None:
    media_path = _make_tiny_media_segment(tmp_path / "private-segment37.wav")
    manifest = _write_one_segment_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    source_url = "fixture://integration/private.m3u8?token=secret"
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    segment = resolve_segments(manifest, source_url=source_url)[0]
    segment_id = _persist_segment(conn, segment)
    chunk = decode_segment_audio(segment)
    private_phrase = "private transcript phrase"
    word = DeterministicTranscriber([(private_phrase, 0.02, 0.04, 0.5)]).transcribe(chunk).words[0]
    invalid_words = (WordToken(text=word.text, start_ts=word.start_ts, end_ts=word.start_ts - 0.01, confidence=word.confidence),)

    with pytest.raises(ValueError, match="word.end_ts") as exc_info:
        insert_transcript_words(
            conn,
            segment_id=segment_id,
            source_url=chunk.source_url,
            segment_sequence=chunk.segment_sequence,
            words=invalid_words,
        )

    message = str(exc_info.value)
    assert private_phrase not in message
    assert "token=secret" not in message
    assert "private.m3u8" not in message
    assert get_transcript_words_for_segment(conn, segment_id) == ()
