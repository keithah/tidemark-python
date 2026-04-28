from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.audio import AudioDecodeError, decode_segment_audio
from tidemark.ingest import resolve_segments
from tidemark.store import get_segment, insert_segment, migrate


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


def test_resolved_segment_persists_and_decodes_with_matching_audio_context(tmp_path: Path) -> None:
    media_path = _make_tiny_media_segment(tmp_path / "segment37.wav")
    manifest = _write_one_segment_manifest(tmp_path / "playlist.m3u8", media_path.name)
    source_url = "fixture://integration/source.m3u8?token=secret"
    conn = sqlite3.connect(":memory:")
    migrate(conn)

    segments = resolve_segments(manifest, source_url=source_url)
    assert len(segments) == 1
    segment = segments[0]
    row_id = _persist_segment(conn, segment)

    stored = get_segment(conn, row_id)
    chunk = decode_segment_audio(segment)

    assert stored is not None
    assert stored.id == row_id
    assert stored.source_url == segment.source_url == source_url
    assert stored.sequence == segment.sequence == 37
    assert stored.resolved_uri == segment.resolved_uri == media_path.as_uri()
    assert stored.local_path == segment.local_path == str(media_path)
    assert stored.start_ts == segment.start_ts == 0.0
    assert stored.duration_seconds == segment.duration_seconds == pytest.approx(0.20)
    assert stored.byte_length == segment.byte_length == media_path.stat().st_size
    assert stored.sha256 == segment.sha256
    assert stored.metadata == segment.metadata

    assert chunk.pcm_bytes
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.sample_format == "s16le"
    assert chunk.segment_sequence == stored.sequence
    assert chunk.source_url == stored.source_url
    assert chunk.resolved_uri == stored.resolved_uri
    assert chunk.start_ts == stored.start_ts
    assert chunk.duration_seconds == stored.duration_seconds
    assert chunk.byte_length == len(chunk.pcm_bytes)
    assert chunk.metadata == {}


def test_decode_failure_after_persistence_is_redacted_and_leaves_segment_row_inspectable(tmp_path: Path) -> None:
    media_path = _make_tiny_media_segment(tmp_path / "private-segment37.wav")
    manifest = _write_one_segment_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    source_url = "fixture://integration/private.m3u8?token=secret"
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment = resolve_segments(manifest, source_url=source_url)[0]
    row_id = _persist_segment(conn, segment)

    media_path.write_bytes(b"corrupted private media bytes with token=secret")

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(
            media_path,
            source_url=segment.source_url,
            sequence=segment.sequence,
            resolved_uri=segment.resolved_uri,
            start_ts=segment.start_ts,
            duration_seconds=segment.duration_seconds,
        )

    message = str(excinfo.value)
    assert "Audio decode error during decode at sequence 37" in message
    assert "ffmpeg returned non-zero status" in message
    assert "token=secret" not in message
    assert "corrupted private media bytes" not in message
    assert "private-segment37" not in message
    assert "private-playlist" not in message
    assert "Invalid data" not in message

    stored = get_segment(conn, row_id)
    assert stored is not None
    assert stored.id == row_id
    assert stored.source_url == source_url
    assert stored.sequence == 37
    assert stored.resolved_uri == segment.resolved_uri
    assert stored.local_path == str(media_path)
    assert stored.byte_length == segment.byte_length
    assert stored.sha256 == segment.sha256
    assert media_path.stat().st_size != stored.byte_length
