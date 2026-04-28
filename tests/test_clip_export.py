from __future__ import annotations

import hashlib
import sqlite3
import wave
from pathlib import Path

import pytest

from tidemark.clip import (
    ClipCoverageMissing,
    ClipDatabaseMissing,
    ClipExportError,
    ClipExportResult,
    ClipWriteError,
    MalformedClipRequest,
    RetainedAudioInvalid,
    RetainedAudioMissing,
    export_clip,
    export_clip_db,
)
from tidemark.store import find_retained_audio_covering, insert_retained_audio, insert_segment, migrate


def _pcm_s16le(frames: int, *, start: int = 0, channels: int = 1) -> bytes:
    samples: list[bytes] = []
    for frame in range(frames):
        for channel in range(channels):
            value = ((start + frame + channel) % 128) - 64
            samples.append(value.to_bytes(2, "little", signed=True))
    return b"".join(samples)


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int = 10, channels: int = 1, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _wav_metadata(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _db(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    db_path = tmp_path / "private-state" / "tidemark.sqlite3"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    migrate(conn)
    return db_path, conn


def _insert_retained(
    conn: sqlite3.Connection,
    *,
    path: str,
    start_ts: float = 10.0,
    duration_seconds: float = 10.0,
    source_url: str = "https://example.test/private/source.m3u8?token=secret",
    sequence: int = 1,
    sample_rate: int = 10,
    channels: int = 1,
    format: str = "wav",
    sample_format: str = "s16le",
    byte_length: int,
    sha256: str,
) -> int:
    segment_id = insert_segment(
        conn,
        source_url=source_url,
        sequence=sequence,
        resolved_uri="file:///Users/alice/private/segment.ts?token=secret",
        local_path="/Users/alice/private/segment.ts",
        start_ts=start_ts,
        duration_seconds=duration_seconds,
        byte_length=byte_length,
        sha256="0" * 64,
    )
    return insert_retained_audio(
        conn,
        segment_id=segment_id,
        source_url=source_url,
        segment_sequence=sequence,
        path=path,
        format=format,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        start_ts=start_ts,
        duration_seconds=duration_seconds,
        byte_length=byte_length,
        sha256=sha256,
    )


def _retained_fixture(tmp_path: Path, *, relative: bool = False, frames: int = 100) -> tuple[Path, sqlite3.Connection, Path, bytes]:
    db_path, conn = _db(tmp_path)
    retained_dir = db_path.parent / "audio"
    retained_dir.mkdir()
    retained_path = retained_dir / "secret-fixture-name.wav"
    pcm = _pcm_s16le(frames)
    _write_wav(retained_path, pcm)
    byte_length, sha256 = _wav_metadata(retained_path)
    stored_path = retained_path.relative_to(db_path.parent).as_posix() if relative else str(retained_path)
    _insert_retained(
        conn,
        path=stored_path,
        start_ts=10.0,
        duration_seconds=10.0,
        byte_length=byte_length,
        sha256=sha256,
    )
    return db_path, conn, retained_path, pcm


def _assert_redacted(exc: BaseException, *sensitive: str) -> None:
    message = str(exc)
    assert message
    for value in sensitive:
        assert value not in message
    assert "secret" not in message
    assert "example.test" not in message
    assert "Users/alice" not in message


def test_find_retained_audio_covering_validates_timestamp_and_chooses_recent_narrow_row(tmp_path: Path) -> None:
    db_path, conn = _db(tmp_path)
    wav_path = db_path.parent / "a.wav"
    _write_wav(wav_path, _pcm_s16le(100))
    byte_length, sha256 = _wav_metadata(wav_path)
    _insert_retained(conn, path=str(wav_path), start_ts=0, duration_seconds=30, sequence=1, byte_length=byte_length, sha256=sha256)
    expected_id = _insert_retained(conn, path=str(wav_path), start_ts=8, duration_seconds=4, sequence=2, byte_length=byte_length, sha256=sha256)
    _insert_retained(conn, path=str(wav_path), start_ts=8, duration_seconds=8, sequence=3, byte_length=byte_length, sha256=sha256)

    assert find_retained_audio_covering(conn, 10).id == expected_id
    assert find_retained_audio_covering(conn, 8).id == expected_id
    assert find_retained_audio_covering(conn, 12).id == expected_id
    assert find_retained_audio_covering(conn, 99) is None
    with pytest.raises(TypeError, match="at_seconds"):
        find_retained_audio_covering(conn, "10")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at_seconds"):
        find_retained_audio_covering(conn, -0.01)


def test_export_clip_writes_real_wav_with_boundary_clipping_and_metadata(tmp_path: Path) -> None:
    db_path, conn, _retained_path, pcm = _retained_fixture(tmp_path)
    out_path = tmp_path / "clip.wav"

    result = export_clip(conn, db_path=db_path, at_seconds=11.0, context_seconds=2.0, out_path=out_path)

    assert isinstance(result, ClipExportResult)
    assert result.start_ts == pytest.approx(10.0)
    assert result.end_ts == pytest.approx(13.0)
    assert result.duration_seconds == pytest.approx(3.0)
    assert result.sample_rate == 10
    assert result.channels == 1
    assert result.sample_format == "s16le"
    assert result.byte_length == out_path.stat().st_size
    assert result.sha256 == hashlib.sha256(out_path.read_bytes()).hexdigest()
    with wave.open(str(out_path), "rb") as wav:
        assert wav.getframerate() == 10
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 30
        assert wav.readframes(30) == pcm[: 30 * 2]


def test_export_clip_supports_zero_context_and_relative_retained_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path, conn, _retained_path, _pcm = _retained_fixture(tmp_path, relative=True)
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    out_path = tmp_path / "zero.wav"

    result = export_clip(conn, db_path=db_path, at_seconds=15.0, context_seconds=0, out_path=out_path)

    assert result.start_ts == pytest.approx(15.0)
    assert result.end_ts == pytest.approx(15.0)
    assert result.duration_seconds == pytest.approx(0.0)
    with wave.open(str(out_path), "rb") as wav:
        assert wav.getnframes() == 0


def test_export_clip_db_opens_existing_database_and_reports_missing_database(tmp_path: Path) -> None:
    db_path, _conn, _retained_path, _pcm = _retained_fixture(tmp_path)
    out_path = tmp_path / "from-db.wav"

    result = export_clip_db(db_path, at_seconds=19.5, context_seconds=1.0, out_path=out_path)

    assert result.start_ts == pytest.approx(18.5)
    assert result.end_ts == pytest.approx(20.0)
    assert out_path.exists()
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ClipDatabaseMissing) as excinfo:
        export_clip_db(missing, at_seconds=10, context_seconds=1, out_path=tmp_path / "missing.wav")
    _assert_redacted(excinfo.value, str(missing), missing.name)


def test_export_clip_rejects_malformed_inputs_before_db_or_file_reads(tmp_path: Path) -> None:
    db_path, conn, _retained_path, _pcm = _retained_fixture(tmp_path)
    with pytest.raises(TypeError, match="at_seconds"):
        export_clip(conn, db_path=db_path, at_seconds="10", context_seconds=1, out_path=tmp_path / "x.wav")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at_seconds"):
        export_clip(conn, db_path=db_path, at_seconds=-1, context_seconds=1, out_path=tmp_path / "x.wav")
    with pytest.raises(ValueError, match="context_seconds"):
        export_clip(conn, db_path=db_path, at_seconds=10, context_seconds=-1, out_path=tmp_path / "x.wav")
    with pytest.raises(ClipWriteError, match="clip output write failed"):
        export_clip(conn, db_path=db_path, at_seconds=10, context_seconds=1, out_path=tmp_path)


def test_export_clip_reports_missing_coverage_and_retained_file_without_leaking_paths(tmp_path: Path) -> None:
    db_path, conn, retained_path, _pcm = _retained_fixture(tmp_path)
    with pytest.raises(ClipCoverageMissing, match="no retained audio covers timestamp") as coverage:
        export_clip(conn, db_path=db_path, at_seconds=99, context_seconds=1, out_path=tmp_path / "coverage.wav")
    _assert_redacted(coverage.value, str(db_path), db_path.name, str(retained_path), retained_path.name)

    retained_path.unlink()
    with pytest.raises(RetainedAudioMissing) as missing:
        export_clip(conn, db_path=db_path, at_seconds=10, context_seconds=1, out_path=tmp_path / "missing-retained.wav")
    _assert_redacted(missing.value, str(db_path), db_path.name, str(retained_path), retained_path.name)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("byte_length", "byte length"),
        ("sha256", "sha256"),
        ("format", "format"),
        ("sample_format", "sample_format"),
        ("wav_header", "wav"),
    ],
)
def test_export_clip_validates_retained_metadata_and_wav_integrity(
    tmp_path: Path, mutation: str, message: str
) -> None:
    db_path, conn, retained_path, _pcm = _retained_fixture(tmp_path)
    if mutation == "byte_length":
        conn.execute("UPDATE retained_audio SET byte_length = byte_length + 1")
    elif mutation == "sha256":
        conn.execute("UPDATE retained_audio SET sha256 = ?", ("1" * 64,))
    elif mutation == "format":
        conn.execute("UPDATE retained_audio SET format = 'mp3'")
    elif mutation == "sample_format":
        conn.execute("UPDATE retained_audio SET sample_format = 'f32le'")
    elif mutation == "wav_header":
        retained_path.write_bytes(b"not a wav file")
        conn.execute(
            "UPDATE retained_audio SET byte_length = ?, sha256 = ?",
            (retained_path.stat().st_size, hashlib.sha256(retained_path.read_bytes()).hexdigest()),
        )
    conn.commit()

    with pytest.raises(RetainedAudioInvalid, match=message) as excinfo:
        export_clip(conn, db_path=db_path, at_seconds=10, context_seconds=1, out_path=tmp_path / f"{mutation}.wav")
    _assert_redacted(excinfo.value, str(db_path), db_path.name, str(retained_path), retained_path.name)


def test_export_clip_wraps_database_read_and_output_write_failures(tmp_path: Path) -> None:
    db_path, conn, _retained_path, _pcm = _retained_fixture(tmp_path)
    conn.close()
    with pytest.raises(ClipExportError, match="database read failed during clip export"):
        export_clip(conn, db_path=db_path, at_seconds=10, context_seconds=1, out_path=tmp_path / "closed.wav")

    db_path, conn, _retained_path, _pcm = _retained_fixture(tmp_path / "write")
    out_dir = tmp_path / "collision.wav"
    out_dir.mkdir()
    with pytest.raises(ClipWriteError, match="clip output write failed"):
        export_clip(conn, db_path=db_path, at_seconds=10, context_seconds=1, out_path=out_dir)
