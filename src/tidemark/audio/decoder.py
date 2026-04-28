"""ffmpeg-backed audio decode boundary."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg

from tidemark.audio.models import AudioChunk
from tidemark.ingest.segments import SegmentRecord

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_FORMAT = "s16le"
DEFAULT_DECODE_TIMEOUT_SECONDS = 30.0


class AudioDecodeError(ValueError):
    """Redacted audio decode failure with phase and sequence context only."""


def _error(phase: str, *, sequence: int | None, detail: str) -> AudioDecodeError:
    context = f" during {phase}"
    if sequence is not None:
        context += f" at sequence {sequence}"
    return AudioDecodeError(f"Audio decode error{context}: {detail}")


@dataclass(frozen=True)
class _DecodeInput:
    data: bytes | None
    path: Path | None
    source_url: str
    sequence: int
    resolved_uri: str
    start_ts: float
    duration_seconds: float | None
    metadata: dict[str, str]


def decode_segment_audio(
    segment: SegmentRecord | bytes | str | Path,
    *,
    source_url: str = "",
    sequence: int = 0,
    resolved_uri: str | None = None,
    start_ts: float = 0.0,
    duration_seconds: float | None = None,
    metadata: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_DECODE_TIMEOUT_SECONDS,
) -> AudioChunk:
    """Decode a segment or equivalent media payload to mono 16 kHz PCM bytes."""
    decode_input = _coerce_decode_input(
        segment,
        source_url=source_url,
        sequence=sequence,
        resolved_uri=resolved_uri,
        start_ts=start_ts,
        duration_seconds=duration_seconds,
        metadata=metadata,
    )

    ffmpeg = _ffmpeg_executable(sequence=decode_input.sequence)
    command = _build_ffmpeg_command(ffmpeg)
    stdin_data = _media_bytes(decode_input)

    try:
        result = subprocess.run(
            command,
            input=stdin_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise _error("decode", sequence=decode_input.sequence, detail="ffmpeg timed out") from exc
    except OSError as exc:
        raise _error("decode", sequence=decode_input.sequence, detail="ffmpeg execution failed") from exc

    if result.returncode != 0:
        cause = RuntimeError(_decode_stderr(result.stderr))
        raise _error("decode", sequence=decode_input.sequence, detail="ffmpeg returned non-zero status") from cause

    pcm = result.stdout
    if not pcm:
        raise _error("decode", sequence=decode_input.sequence, detail="ffmpeg produced no audio")

    return AudioChunk(
        pcm_bytes=pcm,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        sample_format=DEFAULT_SAMPLE_FORMAT,
        segment_sequence=decode_input.sequence,
        source_url=decode_input.source_url,
        resolved_uri=decode_input.resolved_uri,
        start_ts=decode_input.start_ts,
        duration_seconds=decode_input.duration_seconds,
        byte_length=len(pcm),
        metadata=decode_input.metadata,
    )


def _coerce_decode_input(
    segment: SegmentRecord | bytes | str | Path,
    *,
    source_url: str,
    sequence: int,
    resolved_uri: str | None,
    start_ts: float,
    duration_seconds: float | None,
    metadata: dict[str, str] | None,
) -> _DecodeInput:
    if isinstance(segment, SegmentRecord):
        try:
            data = segment.load_bytes()
        except Exception as exc:
            raise _error("load", sequence=segment.sequence, detail="no segment bytes available") from exc
        return _DecodeInput(
            data=data,
            path=None,
            source_url=segment.source_url,
            sequence=segment.sequence,
            resolved_uri=segment.resolved_uri,
            start_ts=segment.start_ts,
            duration_seconds=segment.duration_seconds,
            metadata=_public_metadata(segment.metadata),
        )

    if isinstance(segment, bytes):
        return _DecodeInput(
            data=segment,
            path=None,
            source_url=source_url,
            sequence=sequence,
            resolved_uri=resolved_uri or "",
            start_ts=float(start_ts),
            duration_seconds=duration_seconds,
            metadata=_public_metadata(metadata),
        )

    path = Path(segment)
    return _DecodeInput(
        data=None,
        path=path,
        source_url=source_url,
        sequence=sequence,
        resolved_uri=resolved_uri or path.resolve().as_uri(),
        start_ts=float(start_ts),
        duration_seconds=duration_seconds,
        metadata=_public_metadata(metadata),
    )


def _media_bytes(decode_input: _DecodeInput) -> bytes:
    if decode_input.data is not None:
        return decode_input.data
    if decode_input.path is None:
        raise _error("load", sequence=decode_input.sequence, detail="no segment bytes available")
    try:
        return decode_input.path.read_bytes()
    except Exception as exc:
        raise _error("load", sequence=decode_input.sequence, detail="no segment bytes available") from exc


def _ffmpeg_executable(*, sequence: int) -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise _error("ffmpeg", sequence=sequence, detail="ffmpeg unavailable") from exc


def _build_ffmpeg_command(ffmpeg: str) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        str(DEFAULT_CHANNELS),
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-f",
        DEFAULT_SAMPLE_FORMAT,
        "pipe:1",
    ]


def _public_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    if not metadata:
        return {}
    allowed_keys = {"source_label", "program", "variant", "rendition", "language"}
    return {str(key): str(value) for key, value in metadata.items() if key in allowed_keys}


def _decode_stderr(stderr: bytes | str | None) -> str:
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace")
    return stderr


def audio_sha256(chunk: AudioChunk) -> str:
    """Return a deterministic digest for tests or future internal diagnostics."""
    return hashlib.sha256(chunk.pcm_bytes).hexdigest()


setattr(decode_segment_audio, "build_ffmpeg_command", _build_ffmpeg_command)
