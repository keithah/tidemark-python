"""Retained-audio clip export helpers.

The clip boundary accepts schema-v4 retained-audio metadata and writes a bounded
WAV clip without surfacing database paths, retained paths, source URLs, or raw
exception text in public errors.
"""

from __future__ import annotations

import hashlib
import sqlite3
import wave
from dataclasses import dataclass
from numbers import Real
from os import PathLike
from pathlib import Path

from tidemark.store import RetainedAudioStoreRecord, connect_db, find_retained_audio_covering

_BYTES_PER_S16LE_SAMPLE = 2
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ClipExportResult:
    """Metadata for an exported retained-audio WAV clip."""

    path: Path
    start_ts: float
    end_ts: float
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_format: str
    byte_length: int
    sha256: str


class ClipExportError(Exception):
    """Base class for redacted clip export failures."""


class MalformedClipRequest(ClipExportError, ValueError):
    """The caller supplied malformed clip export inputs."""


class ClipDatabaseMissing(ClipExportError, FileNotFoundError):
    """The requested SQLite database path does not exist."""


class ClipCoverageMissing(ClipExportError):
    """No retained-audio row covers the requested timestamp."""


class RetainedAudioMissing(ClipExportError, FileNotFoundError):
    """A retained-audio metadata row points at a missing file."""


class RetainedAudioInvalid(ClipExportError):
    """Retained-audio metadata or WAV bytes failed integrity validation."""


class ClipWriteError(ClipExportError, OSError):
    """The requested output WAV could not be written."""


def export_clip_db(
    path: str | PathLike[str],
    *,
    at_seconds: int | float,
    context_seconds: int | float,
    out_path: str | PathLike[str],
) -> ClipExportResult:
    """Open a SQLite database and export a retained-audio clip."""
    db_path = _normalize_path("path", path)
    _validate_export_request(at_seconds=at_seconds, context_seconds=context_seconds, out_path=out_path)
    if not db_path.exists() or not db_path.is_file():
        raise ClipDatabaseMissing("clip database is missing")

    conn = connect_db(db_path)
    try:
        return export_clip(conn, db_path=db_path, at_seconds=at_seconds, context_seconds=context_seconds, out_path=out_path)
    finally:
        conn.close()


def export_clip(
    conn: sqlite3.Connection,
    *,
    db_path: str | PathLike[str],
    at_seconds: int | float,
    context_seconds: int | float,
    out_path: str | PathLike[str],
) -> ClipExportResult:
    """Export a bounded WAV clip from the retained-audio row covering ``at_seconds``."""
    database_path = _normalize_path("db_path", db_path)
    output_path = _validate_export_request(
        at_seconds=at_seconds,
        context_seconds=context_seconds,
        out_path=out_path,
    )
    normalized_at = _require_non_negative_number("at_seconds", at_seconds)
    normalized_context = _require_non_negative_number("context_seconds", context_seconds)

    row = _find_covering_row(conn, normalized_at)
    retained_path = _resolve_retained_path(row, database_path)
    _validate_retained_metadata(row)
    _validate_retained_file_bytes(row, retained_path)

    clip_start = max(row.start_ts, normalized_at - normalized_context)
    clip_end = min(row.start_ts + row.duration_seconds, normalized_at + normalized_context)
    if clip_end < clip_start:
        raise RetainedAudioInvalid("invalid retained audio timing metadata")

    try:
        return _write_clip(row, retained_path=retained_path, output_path=output_path, clip_start=clip_start, clip_end=clip_end)
    except ClipExportError:
        raise
    except Exception as exc:
        _remove_partial(output_path)
        raise ClipWriteError("clip output write failed") from exc


def _find_covering_row(conn: sqlite3.Connection, at_seconds: float) -> RetainedAudioStoreRecord:
    try:
        row = find_retained_audio_covering(conn, at_seconds)
    except sqlite3.Error as exc:
        raise ClipExportError("database read failed during clip export") from exc
    except (TypeError, ValueError) as exc:
        raise RetainedAudioInvalid("invalid retained audio metadata") from exc
    if row is None:
        raise ClipCoverageMissing("no retained audio covers timestamp")
    return row


def _validate_export_request(
    *,
    at_seconds: int | float,
    context_seconds: int | float,
    out_path: str | PathLike[str],
) -> Path:
    _require_non_negative_number("at_seconds", at_seconds)
    _require_non_negative_number("context_seconds", context_seconds)
    output_path = _normalize_path("out_path", out_path)
    if output_path.exists() and output_path.is_dir():
        raise ClipWriteError("clip output write failed")
    return output_path


def _normalize_path(name: str, value: str | PathLike[str]) -> Path:
    if isinstance(value, (str, PathLike)):
        return Path(value)
    raise MalformedClipRequest(f"{name} must be a filesystem path")


def _require_non_negative_number(name: str, value: int | float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative number")
    normalized = float(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


def _resolve_retained_path(row: RetainedAudioStoreRecord, db_path: Path) -> Path:
    if not isinstance(row.path, str) or not row.path.strip():
        raise RetainedAudioInvalid("invalid retained audio path metadata")
    retained_path = Path(row.path)
    if retained_path.is_absolute():
        return retained_path
    return db_path.parent / retained_path


def _validate_retained_metadata(row: RetainedAudioStoreRecord) -> None:
    if row.format != "wav":
        raise RetainedAudioInvalid("invalid retained audio format metadata")
    if row.sample_format != "s16le":
        raise RetainedAudioInvalid("invalid retained audio sample_format metadata")
    if not isinstance(row.sample_rate, int) or row.sample_rate <= 0:
        raise RetainedAudioInvalid("invalid retained audio sample_rate metadata")
    if not isinstance(row.channels, int) or row.channels <= 0:
        raise RetainedAudioInvalid("invalid retained audio channels metadata")
    if row.start_ts < 0 or row.duration_seconds < 0:
        raise RetainedAudioInvalid("invalid retained audio timing metadata")
    if row.byte_length < 0:
        raise RetainedAudioInvalid("invalid retained audio byte length metadata")
    if not isinstance(row.sha256, str) or len(row.sha256) != 64:
        raise RetainedAudioInvalid("invalid retained audio sha256 metadata")


def _validate_retained_file_bytes(row: RetainedAudioStoreRecord, retained_path: Path) -> None:
    if not retained_path.exists() or not retained_path.is_file():
        raise RetainedAudioMissing("retained audio file is missing")
    try:
        byte_length, digest = _hash_file(retained_path)
    except Exception as exc:
        raise RetainedAudioInvalid("retained audio integrity check failed") from exc
    if byte_length != row.byte_length:
        raise RetainedAudioInvalid("retained audio byte length mismatch")
    if digest != row.sha256.lower():
        raise RetainedAudioInvalid("retained audio sha256 mismatch")


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _write_clip(
    row: RetainedAudioStoreRecord,
    *,
    retained_path: Path,
    output_path: Path,
    clip_start: float,
    clip_end: float,
) -> ClipExportResult:
    try:
        with wave.open(str(retained_path), "rb") as source:
            _validate_wav_header(row, source)
            start_frame = _timestamp_to_frame(row, clip_start)
            end_frame = _timestamp_to_frame(row, clip_end)
            frame_count = max(0, end_frame - start_frame)
            source.setpos(start_frame)
            frames = source.readframes(frame_count)
            expected_bytes = frame_count * row.channels * _BYTES_PER_S16LE_SAMPLE
            if len(frames) != expected_bytes:
                raise RetainedAudioInvalid("retained wav frame data is unreadable")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as target:
                target.setnchannels(source.getnchannels())
                target.setsampwidth(source.getsampwidth())
                target.setframerate(source.getframerate())
                target.writeframes(frames)
    except RetainedAudioInvalid:
        _remove_partial(output_path)
        raise
    except (wave.Error, EOFError) as exc:
        _remove_partial(output_path)
        raise RetainedAudioInvalid("retained wav header is invalid") from exc
    except Exception as exc:
        _remove_partial(output_path)
        raise ClipWriteError("clip output write failed") from exc

    try:
        byte_length, digest = _hash_file(output_path)
    except Exception as exc:
        _remove_partial(output_path)
        raise ClipWriteError("clip output write failed") from exc

    return ClipExportResult(
        path=output_path,
        start_ts=clip_start,
        end_ts=clip_end,
        duration_seconds=clip_end - clip_start,
        sample_rate=row.sample_rate,
        channels=row.channels,
        sample_format=row.sample_format,
        byte_length=byte_length,
        sha256=digest,
    )


def _validate_wav_header(row: RetainedAudioStoreRecord, source: wave.Wave_read) -> None:
    if source.getframerate() != row.sample_rate:
        raise RetainedAudioInvalid("retained wav sample_rate mismatch")
    if source.getnchannels() != row.channels:
        raise RetainedAudioInvalid("retained wav channels mismatch")
    if source.getsampwidth() != _BYTES_PER_S16LE_SAMPLE:
        raise RetainedAudioInvalid("retained wav sample_width mismatch")
    required_frames = _timestamp_to_frame(row, row.start_ts + row.duration_seconds)
    if source.getnframes() < required_frames:
        raise RetainedAudioInvalid("retained wav duration metadata mismatch")


def _timestamp_to_frame(row: RetainedAudioStoreRecord, timestamp: float) -> int:
    relative = max(0.0, timestamp - row.start_ts)
    return round(relative * row.sample_rate)


def _remove_partial(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass
