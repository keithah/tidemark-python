"""Retained audio file writer for fingerprint workflows."""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from tidemark.audio import AudioChunk

_BYTES_PER_S16LE_SAMPLE = 2


@dataclass(frozen=True)
class RetainedAudioFile:
    """Metadata for a retained WAV file written beside a SQLite database."""

    path: Path
    format: str
    sample_rate: int
    channels: int
    sample_format: str
    start_ts: float
    duration_seconds: float
    byte_length: int
    sha256: str


class RetentionError(ValueError):
    """Redacted retention failure with phase and sequence context only."""


def write_retained_audio(
    chunk: AudioChunk,
    *,
    db_path: str | Path,
    retention_dir: str | Path | None = None,
) -> RetainedAudioFile:
    """Write an ``AudioChunk`` as a retained WAV and return file metadata.

    File names are generated from segment sequence and the WAV-file digest so
    callers never embed source URLs, local source paths, or PCM payloads in the
    retained path. The returned digest and byte length describe the written WAV
    container, not the raw PCM bytes.
    """
    duration_seconds = _validate_chunk(chunk)
    target_dir = _target_directory(db_path, retention_dir=retention_dir)
    sequence = chunk.segment_sequence

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise _error("directory", sequence=sequence, detail="could not prepare retention directory") from exc

    try:
        wav_bytes = _render_wav_bytes(chunk)
        digest = hashlib.sha256(wav_bytes).hexdigest()
        path = target_dir / f"segment-{sequence}-{digest[:12]}.wav"
        path.write_bytes(wav_bytes)
    except RetentionError:
        raise
    except Exception as exc:
        raise _error("write", sequence=sequence, detail="could not write retained wav") from exc

    try:
        written = path.read_bytes()
    except Exception as exc:
        raise _error("write", sequence=sequence, detail="could not verify retained wav") from exc

    return RetainedAudioFile(
        path=path,
        format="wav",
        sample_rate=chunk.sample_rate,
        channels=chunk.channels,
        sample_format=chunk.sample_format,
        start_ts=float(chunk.start_ts),
        duration_seconds=duration_seconds,
        byte_length=len(written),
        sha256=hashlib.sha256(written).hexdigest(),
    )


def _target_directory(db_path: str | Path, *, retention_dir: str | Path | None) -> Path:
    if retention_dir is not None:
        return Path(retention_dir)
    database = Path(db_path)
    return database.parent / f"{database.stem}-audio"


def _render_wav_bytes(chunk: AudioChunk) -> bytes:
    import io

    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(chunk.channels)
            wav.setsampwidth(_BYTES_PER_S16LE_SAMPLE)
            wav.setframerate(chunk.sample_rate)
            wav.writeframes(chunk.pcm_bytes)
    except Exception as exc:
        raise _error("write", sequence=chunk.segment_sequence, detail="could not encode retained wav") from exc
    return buffer.getvalue()


def _validate_chunk(chunk: AudioChunk) -> float:
    sequence = chunk.segment_sequence
    if not isinstance(chunk.sample_rate, int) or isinstance(chunk.sample_rate, bool) or chunk.sample_rate <= 0:
        raise _error("validation", sequence=sequence, detail="invalid sample_rate")
    if not isinstance(chunk.channels, int) or isinstance(chunk.channels, bool) or chunk.channels <= 0:
        raise _error("validation", sequence=sequence, detail="invalid channels")
    if chunk.sample_format != "s16le":
        raise _error("validation", sequence=sequence, detail="invalid sample_format")
    if not isinstance(chunk.pcm_bytes, bytes) or len(chunk.pcm_bytes) == 0:
        raise _error("validation", sequence=sequence, detail="invalid pcm_bytes")
    frame_width = chunk.channels * _BYTES_PER_S16LE_SAMPLE
    if len(chunk.pcm_bytes) % frame_width != 0:
        raise _error("validation", sequence=sequence, detail="invalid pcm_bytes")
    if not _is_number(chunk.start_ts) or chunk.start_ts < 0:
        raise _error("validation", sequence=sequence, detail="invalid start_ts")

    if chunk.duration_seconds is None:
        return len(chunk.pcm_bytes) / (chunk.sample_rate * chunk.channels * _BYTES_PER_S16LE_SAMPLE)
    if not _is_number(chunk.duration_seconds) or chunk.duration_seconds < 0:
        raise _error("validation", sequence=sequence, detail="invalid duration_seconds")
    return float(chunk.duration_seconds)


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _error(phase: str, *, sequence: int, detail: str) -> RetentionError:
    return RetentionError(f"Retention error during {phase} at sequence {sequence}: {detail}")
