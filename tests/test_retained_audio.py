from __future__ import annotations

import hashlib
import sqlite3
import wave
from pathlib import Path

import pytest

from tidemark.audio import AudioChunk
from tidemark.fingerprint import RetainedAudioFile, RetentionError, write_retained_audio
from tidemark.store import get_retained_audio, insert_retained_audio, insert_segment, migrate


def _pcm_s16le(frames: int = 800) -> bytes:
    # Alternating low-amplitude signed 16-bit samples; valid mono s16le PCM.
    return b"".join(((i % 64) - 32).to_bytes(2, "little", signed=True) for i in range(frames))


def _chunk(
    *,
    pcm_bytes: bytes | None = None,
    sample_rate: int = 8000,
    channels: int = 1,
    sample_format: str = "s16le",
    segment_sequence: int = 7,
    source_url: str = "https://example.test/private/stream.m3u8?token=secret",
    resolved_uri: str = "file:///Users/alice/private/segment.wav?token=secret",
    duration_seconds: float | None = 0.1,
) -> AudioChunk:
    payload = _pcm_s16le() if pcm_bytes is None else pcm_bytes
    return AudioChunk(
        pcm_bytes=payload,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        segment_sequence=segment_sequence,
        source_url=source_url,
        resolved_uri=resolved_uri,
        start_ts=12.5,
        duration_seconds=duration_seconds,
        byte_length=len(payload),
        metadata={"source_label": "fixture", "private_path": "/Users/alice/private"},
    )


def test_write_retained_audio_writes_deterministic_wav_beside_db_and_stores_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "tidemark.sqlite3"
    db_path.parent.mkdir()
    chunk = _chunk(duration_seconds=None)

    retained = write_retained_audio(chunk, db_path=db_path)

    expected_dir = db_path.parent / "tidemark-audio"
    assert isinstance(retained, RetainedAudioFile)
    assert retained.path.parent == expected_dir
    assert retained.path.name.startswith("segment-7-")
    assert retained.path.name.endswith(".wav")
    assert "token" not in retained.path.name
    assert "private" not in retained.path.name
    assert retained.format == "wav"
    assert retained.sample_rate == 8000
    assert retained.channels == 1
    assert retained.sample_format == "s16le"
    assert retained.start_ts == 12.5
    assert retained.duration_seconds == pytest.approx(0.1)
    assert retained.byte_length == retained.path.stat().st_size
    assert retained.sha256 == hashlib.sha256(retained.path.read_bytes()).hexdigest()

    with wave.open(str(retained.path), "rb") as wav:
        assert wav.getframerate() == 8000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 800
        assert wav.readframes(800) == chunk.pcm_bytes

    conn = sqlite3.connect(db_path)
    migrate(conn)
    segment_id = insert_segment(
        conn,
        source_url=chunk.source_url,
        sequence=chunk.segment_sequence,
        resolved_uri=chunk.resolved_uri,
        local_path="/Users/alice/private/original.ts",
        start_ts=chunk.start_ts,
        duration_seconds=retained.duration_seconds,
        byte_length=chunk.byte_length,
        sha256=hashlib.sha256(chunk.pcm_bytes).hexdigest(),
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

    stored = get_retained_audio(conn, retained_id)
    assert stored is not None
    assert stored.path == str(retained.path)
    assert stored.byte_length == retained.byte_length
    assert stored.sha256 == retained.sha256
    assert stored.duration_seconds == pytest.approx(retained.duration_seconds)


def test_write_retained_audio_uses_safe_caller_retention_dir_and_is_repeatable(tmp_path: Path) -> None:
    retention_dir = tmp_path / "kept-audio"
    chunk = _chunk(
        segment_sequence=42,
        source_url="file:///Users/alice/private/source.wav?token=secret",
        resolved_uri="file:///Users/alice/private/source.wav?token=secret",
    )

    first = write_retained_audio(chunk, db_path=tmp_path / "private-db.sqlite3", retention_dir=retention_dir)
    second = write_retained_audio(chunk, db_path=tmp_path / "private-db.sqlite3", retention_dir=retention_dir)

    assert first == second
    assert first.path.parent == retention_dir
    assert first.path.name.startswith("segment-42-")
    assert first.path.suffix == ".wav"
    assert "source" not in first.path.name
    assert "Users" not in first.path.name
    assert "token" not in first.path.name
    with wave.open(str(first.path), "rb") as wav:
        assert wav.getnframes() == len(chunk.pcm_bytes) // 2


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"sample_format": "f32le"}, "sample_format"),
        ({"pcm_bytes": b""}, "pcm_bytes"),
        ({"pcm_bytes": b"\x00"}, "pcm_bytes"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"channels": 0}, "channels"),
        ({"duration_seconds": -0.1}, "duration_seconds"),
    ],
)
def test_write_retained_audio_rejects_malformed_chunks_without_leaking_private_values(
    tmp_path: Path, kwargs: dict[str, object], field: str
) -> None:
    with pytest.raises(RetentionError, match=field) as excinfo:
        write_retained_audio(_chunk(**kwargs), db_path=tmp_path / "private" / "tidemark.db")  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "Retention error during validation at sequence 7" in message
    assert "token=secret" not in message
    assert "example.test" not in message
    assert "Users/alice" not in message
    assert "private" not in message


def test_write_retained_audio_wraps_filesystem_failures_without_paths_or_sources(tmp_path: Path) -> None:
    collision = tmp_path / "not-a-directory"
    collision.write_text("collision")

    with pytest.raises(RetentionError) as excinfo:
        write_retained_audio(_chunk(), db_path=tmp_path / "state" / "tidemark.db", retention_dir=collision)

    message = str(excinfo.value)
    assert "Retention error during directory at sequence 7" in message
    assert "not-a-directory" not in message
    assert "tidemark.db" not in message
    assert "token=secret" not in message
    assert "example.test" not in message
    assert "private" not in message
