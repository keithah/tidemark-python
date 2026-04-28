from __future__ import annotations

import hashlib
import importlib
import sqlite3
import subprocess
import sys
import wave
from collections.abc import Iterable
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.audio import decode_segment_audio
from tidemark.fingerprint import FingerprintError, fingerprint_audio_chunk, write_retained_audio
from tidemark.ingest import resolve_segments
from tidemark.store import (
    SCHEMA_VERSION,
    get_fingerprint_cache,
    get_retained_audio,
    get_segment,
    get_song,
    insert_fingerprint_cache,
    insert_retained_audio,
    insert_segment,
    insert_song,
    migrate,
)


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
            "sine=frequency=523.25:duration=0.20:sample_rate=8000",
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


def _write_one_segment_manifest(path: Path, segment_name: str) -> Path:
    path.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-MEDIA-SEQUENCE:41",
                "#EXTINF:0.20,",
                segment_name,
                "#EXT-X-ENDLIST",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _fingerprint_backend(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> str:
    payload = b"".join(pcmiter)
    return f"fp:{sample_rate}:{channels}:{hashlib.sha256(payload).hexdigest()}"


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%' order by name"
        )
    ]


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


def test_decoded_fixture_fingerprints_retains_and_persists_schema_v4_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "acoustid", None)
    assert importlib.import_module("tidemark.fingerprint").fingerprint_audio_chunk is fingerprint_audio_chunk

    media_path = _make_tiny_wav(tmp_path / "private-original-name.wav")
    manifest = _write_one_segment_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    source_url = "fixture://integration/private-source.m3u8?token=secret"
    db_path = tmp_path / "state" / "tidemark.sqlite3"
    db_path.parent.mkdir()

    conn = sqlite3.connect(db_path)
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
    assert _table_names(conn) == [
        "ad_events",
        "fingerprint_cache",
        "retained_audio",
        "segments",
        "songs",
        "transcript_words",
    ]

    segments = resolve_segments(manifest, source_url=source_url)
    assert len(segments) == 1
    segment = segments[0]
    segment_id = _persist_segment(conn, segment)
    stored_segment = get_segment(conn, segment_id)
    chunk = decode_segment_audio(segment)
    fingerprint = fingerprint_audio_chunk(chunk, backend=_fingerprint_backend)
    retained = write_retained_audio(chunk, db_path=db_path)

    song_id = insert_song(
        conn,
        segment_id=segment_id,
        source_url=fingerprint.source_url,
        segment_sequence=fingerprint.segment_sequence,
        start_ts=fingerprint.start_ts,
        duration_seconds=fingerprint.duration_seconds,
        fingerprint=fingerprint.fingerprint,
        acoustid_id="acoustid-fixture-1",
        recording_id="recording-fixture-1",
        title="Fixture tone",
        artist="Tidemark tests",
        album="Generated media",
        score=0.91,
        lookup_source="deterministic-test-backend",
    )
    insert_fingerprint_cache(
        conn,
        fingerprint=fingerprint.fingerprint,
        acoustid_id="acoustid-fixture-1",
        recording_id="recording-fixture-1",
        title="Fixture tone",
        artist="Tidemark tests",
        album="Generated media",
        score=0.91,
        raw_status="ok",
        lookup_source="deterministic-test-backend",
    )
    retained_id = insert_retained_audio(
        conn,
        segment_id=segment_id,
        source_url=chunk.source_url,
        segment_sequence=chunk.segment_sequence,
        path=str(retained.path),
        format=retained.format,
        sample_rate=retained.sample_rate,
        channels=retained.channels,
        sample_format=retained.sample_format,
        start_ts=retained.start_ts,
        duration_seconds=retained.duration_seconds,
        byte_length=retained.byte_length,
        sha256=retained.sha256,
    )

    stored_song = get_song(conn, song_id)
    stored_cache = get_fingerprint_cache(conn, fingerprint.fingerprint)
    stored_retained = get_retained_audio(conn, retained_id)

    assert stored_segment is not None
    assert stored_segment.id == segment_id
    assert stored_segment.source_url == segment.source_url == source_url
    assert stored_segment.sequence == segment.sequence == 41
    assert stored_segment.resolved_uri == segment.resolved_uri == media_path.as_uri()
    assert stored_segment.local_path == str(media_path)
    assert stored_segment.start_ts == segment.start_ts == 0.0
    assert stored_segment.duration_seconds == segment.duration_seconds == pytest.approx(0.20)
    assert stored_segment.byte_length == media_path.stat().st_size
    assert stored_segment.sha256 == hashlib.sha256(media_path.read_bytes()).hexdigest()

    assert chunk.segment_sequence == segment.sequence
    assert chunk.source_url == source_url
    assert chunk.resolved_uri == media_path.as_uri()
    assert chunk.start_ts == pytest.approx(0.0)
    assert chunk.duration_seconds == pytest.approx(0.20)
    assert chunk.pcm_bytes

    assert fingerprint.fingerprint.startswith("fp:16000:1:")
    assert fingerprint.segment_sequence == segment.sequence
    assert fingerprint.source_url == source_url
    assert fingerprint.start_ts == chunk.start_ts
    assert fingerprint.duration_seconds == chunk.duration_seconds

    assert stored_song is not None
    assert stored_song.segment_id == segment_id
    assert stored_song.source_url == source_url
    assert stored_song.segment_sequence == 41
    assert stored_song.start_ts == pytest.approx(0.0)
    assert stored_song.duration_seconds == pytest.approx(0.20)
    assert stored_song.fingerprint == fingerprint.fingerprint
    assert stored_song.lookup_source == "deterministic-test-backend"

    assert stored_cache is not None
    assert stored_cache.fingerprint == fingerprint.fingerprint
    assert stored_cache.acoustid_id == "acoustid-fixture-1"
    assert stored_cache.recording_id == "recording-fixture-1"
    assert stored_cache.title == "Fixture tone"
    assert stored_cache.score == pytest.approx(0.91)
    assert stored_cache.lookup_source == "deterministic-test-backend"

    expected_retention_dir = db_path.parent / "tidemark-audio"
    assert retained.path.parent == expected_retention_dir
    assert retained.path.exists()
    assert retained.path.name.startswith("segment-41-")
    assert retained.path.name.endswith(".wav")
    assert media_path.stem not in retained.path.name
    assert "private" not in retained.path.name
    assert "source" not in retained.path.name
    assert retained.path.read_bytes().startswith(b"RIFF")
    with wave.open(str(retained.path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == len(chunk.pcm_bytes) // 2

    retained_columns = conn.execute("PRAGMA table_info(retained_audio)").fetchall()
    assert all(column[2].upper() != "BLOB" for column in retained_columns)
    assert conn.execute("select typeof(path), typeof(sha256) from retained_audio").fetchone() == ("text", "text")
    assert conn.execute("select count(*) from retained_audio").fetchone()[0] == 1

    assert stored_retained is not None
    assert stored_retained.segment_id == segment_id
    assert stored_retained.source_url == source_url
    assert stored_retained.segment_sequence == 41
    assert stored_retained.path == str(retained.path)
    assert stored_retained.format == "wav"
    assert stored_retained.sample_rate == 16000
    assert stored_retained.channels == 1
    assert stored_retained.sample_format == "s16le"
    assert stored_retained.start_ts == pytest.approx(0.0)
    assert stored_retained.duration_seconds == pytest.approx(0.20)
    assert stored_retained.byte_length == retained.path.stat().st_size
    assert stored_retained.sha256 == hashlib.sha256(retained.path.read_bytes()).hexdigest()


def test_real_acoustid_backend_smoke_skips_when_unavailable(tmp_path: Path) -> None:
    pytest.importorskip("acoustid")
    media_path = _make_tiny_wav(tmp_path / "segment.wav")
    manifest = _write_one_segment_manifest(tmp_path / "playlist.m3u8", media_path.name)
    chunk = decode_segment_audio(resolve_segments(manifest, source_url="fixture://integration/source.m3u8")[0])

    try:
        fingerprint = fingerprint_audio_chunk(chunk)
    except FingerprintError as exc:
        pytest.skip(f"real acoustid/Chromaprint backend unavailable: {exc}")

    assert fingerprint.fingerprint
