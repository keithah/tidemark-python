"""Audio fingerprint generation boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Real
from typing import TypeAlias

from tidemark.audio import AudioChunk
from tidemark.fingerprint.models import AudioFingerprint

FingerprintBackend: TypeAlias = Callable[[int, int, Iterable[bytes]], object]

_BYTES_PER_S16LE_SAMPLE = 2
_PUBLIC_METADATA_KEYS = frozenset({"source_label", "program", "variant", "rendition", "language"})


class FingerprintError(ValueError):
    """Redacted fingerprint failure with phase and sequence context only."""


def fingerprint_audio_chunk(chunk: AudioChunk, *, backend: FingerprintBackend | None = None) -> AudioFingerprint:
    """Generate a Chromaprint fingerprint for a decoded ``AudioChunk``.

    ``backend`` is injectable for tests and future adapters. It receives
    ``sample_rate``, ``channels``, and a one-item bytes iterable containing the
    PCM payload. Return either a fingerprint string or a pyacoustid-compatible
    ``(duration, fingerprint)`` tuple; the tuple duration is accepted for shape
    compatibility but the public model uses the trusted ``AudioChunk`` duration
    or derives it from PCM bytes.
    """
    duration_seconds = _validate_chunk(chunk)
    active_backend = backend or _default_backend(chunk.segment_sequence)

    try:
        raw_result = active_backend(chunk.sample_rate, chunk.channels, (chunk.pcm_bytes,))
    except FingerprintError:
        raise
    except Exception as exc:
        raise _error("backend", sequence=chunk.segment_sequence, detail="backend failed") from exc

    fingerprint = _normalize_backend_result(raw_result, sequence=chunk.segment_sequence)
    return AudioFingerprint(
        fingerprint=fingerprint,
        duration_seconds=duration_seconds,
        algorithm="chromaprint",
        segment_sequence=chunk.segment_sequence,
        source_url=chunk.source_url,
        start_ts=chunk.start_ts,
        metadata=_public_metadata(chunk.metadata),
    )


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

    if chunk.duration_seconds is None:
        return len(chunk.pcm_bytes) / (chunk.sample_rate * chunk.channels * _BYTES_PER_S16LE_SAMPLE)
    if not _is_number(chunk.duration_seconds) or chunk.duration_seconds < 0:
        raise _error("validation", sequence=sequence, detail="invalid duration_seconds")
    return float(chunk.duration_seconds)


def _default_backend(sequence: int) -> FingerprintBackend:
    try:
        import acoustid  # type: ignore[import-not-found]
    except Exception as exc:
        raise _error("dependency", sequence=sequence, detail="acoustid unavailable") from exc

    def _fingerprint(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> object:
        return acoustid.fingerprint(sample_rate, channels, iter(pcmiter))

    return _fingerprint


def _normalize_backend_result(raw_result: object, *, sequence: int) -> str:
    if isinstance(raw_result, (str, bytes)):
        return _validate_fingerprint(raw_result, sequence=sequence)

    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        duration, fingerprint = raw_result
        if not _is_number(duration):
            raise _error("backend", sequence=sequence, detail="malformed fingerprint response")
        if duration < 0:
            raise _error("backend", sequence=sequence, detail="malformed fingerprint response")
        return _validate_fingerprint(fingerprint, sequence=sequence)

    raise _error("backend", sequence=sequence, detail="malformed fingerprint response")


def _validate_fingerprint(value: object, *, sequence: int) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise _error("backend", sequence=sequence, detail="malformed fingerprint field") from exc
    if not isinstance(value, str) or value == "":
        raise _error("backend", sequence=sequence, detail="malformed fingerprint field")
    return value


def _public_metadata(metadata: dict[str, str] | None) -> dict[str, str]:
    if not metadata:
        return {}
    return {str(key): str(value) for key, value in metadata.items() if key in _PUBLIC_METADATA_KEYS}


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _error(phase: str, *, sequence: int, detail: str) -> FingerprintError:
    return FingerprintError(f"Fingerprint error during {phase} at sequence {sequence}: {detail}")
