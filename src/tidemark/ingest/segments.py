"""Local media segment resolution helpers.

This module turns deterministic local fixture inputs into typed segment records for
later persistence/transcription handoff. It deliberately does not decode markers,
call monitor source adapters, or fetch network URLs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse


class SegmentIngestError(ValueError):
    """Redacted segment ingest failure with phase and sequence context only."""


def _error(phase: str, *, sequence: int | None = None, detail: str) -> SegmentIngestError:
    context = f" during {phase}"
    if sequence is not None:
        context += f" at sequence {sequence}"
    return SegmentIngestError(f"Segment ingest error{context}: {detail}")


@dataclass(frozen=True)
class SegmentRecord:
    """Resolved local media segment metadata plus a validating lazy byte loader."""

    source_url: str
    sequence: int
    resolved_uri: str
    local_path: str | None
    start_ts: float
    duration_seconds: float | None
    byte_length: int
    sha256: str
    metadata: dict[str, str] | None = None
    _loader: Callable[[], bytes] | None = None

    def load_bytes(self) -> bytes:
        """Load segment bytes and verify length/hash without leaking content or paths."""
        if self._loader is None:
            raise _error("load", sequence=self.sequence, detail="no byte loader configured")

        try:
            data = self._loader()
        except Exception as exc:
            raise _error("load", sequence=self.sequence, detail="unable to load bytes") from exc

        if not isinstance(data, bytes):
            raise _error("load", sequence=self.sequence, detail="byte loader must return bytes")

        actual_length = len(data)
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_length != self.byte_length:
            raise _error("load", sequence=self.sequence, detail="byte length mismatch")
        if actual_sha256 != self.sha256:
            raise _error("load", sequence=self.sequence, detail="SHA-256 mismatch")
        return data

    def with_loader(self, loader: Callable[[], bytes]) -> SegmentRecord:
        """Return a copy using a replacement byte loader for tests/integration seams."""
        return replace(self, _loader=loader)


def resolve_segments(
    source: str | Path,
    *,
    source_url: str | None = None,
    duration_seconds: float | None = None,
) -> list[SegmentRecord]:
    """Resolve a local HLS manifest or direct local media file into segment records."""
    if _looks_like_network_url(source):
        raise _error("source", detail="unsupported network URL input for local segment ingest")

    path = Path(source)
    if path.suffix.lower() == ".m3u8":
        return resolve_local_hls_segments(path, source_url=source_url)
    return resolve_local_media_file(path, source_url=source_url, duration_seconds=duration_seconds)


def resolve_local_hls_segments(
    manifest_path: str | Path,
    *,
    source_url: str | None = None,
) -> list[SegmentRecord]:
    """Resolve local media segment entries from a deterministic HLS manifest."""
    manifest = Path(manifest_path)
    manifest_uri = _file_uri(manifest)
    effective_source_url = source_url if source_url is not None else manifest_uri

    try:
        manifest_text = manifest.read_text(encoding="utf-8")
    except Exception as exc:
        raise _error("manifest", detail="unable to read manifest") from exc

    lines = manifest_text.splitlines()
    if not any(line.strip() for line in lines):
        raise _error("manifest", detail="empty manifest")

    sequence = 0
    saw_media_segment = False
    pending_duration: float | None = None
    start_ts = 0.0
    records: list[SegmentRecord] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                sequence = _parse_media_sequence(line)
                continue
            if line.startswith("#EXTINF:"):
                if pending_duration is not None:
                    raise _error("manifest", sequence=sequence, detail="missing segment URI after duration")
                pending_duration = _parse_extinf_duration(line, sequence=sequence)
                continue
            continue

        if _looks_like_network_url(line):
            raise _error("segment", sequence=sequence, detail="network segment URI unsupported for local ingest")

        duration = pending_duration
        if duration is None:
            raise _error("manifest", sequence=sequence, detail="missing duration before segment URI")

        segment_path = (manifest.parent / line).resolve()
        records.append(
            _record_for_path(
                segment_path,
                source_url=effective_source_url,
                sequence=sequence,
                resolved_uri=_file_uri(segment_path),
                start_ts=start_ts,
                duration_seconds=duration,
                metadata={"manifest_path": str(manifest), "manifest_uri": manifest_uri},
            )
        )
        saw_media_segment = True
        start_ts += duration
        sequence += 1
        pending_duration = None

    if pending_duration is not None:
        raise _error("manifest", sequence=sequence, detail="missing segment URI after duration")
    if not saw_media_segment:
        raise _error("manifest", detail="manifest contains no media segments")

    return records


def resolve_local_media_file(
    media_path: str | Path,
    *,
    source_url: str | None = None,
    duration_seconds: float | None = None,
) -> list[SegmentRecord]:
    """Resolve a direct local media file as a single sequence-zero segment."""
    if duration_seconds is not None and duration_seconds < 0:
        raise _error("source", sequence=0, detail="duration_seconds must be >= 0")

    path = Path(media_path).resolve()
    resolved_uri = _file_uri(path)
    return [
        _record_for_path(
            path,
            source_url=source_url if source_url is not None else resolved_uri,
            sequence=0,
            resolved_uri=resolved_uri,
            start_ts=0.0,
            duration_seconds=duration_seconds,
            metadata=None,
        )
    ]


def _record_for_path(
    path: Path,
    *,
    source_url: str,
    sequence: int,
    resolved_uri: str,
    start_ts: float,
    duration_seconds: float | None,
    metadata: dict[str, str] | None,
) -> SegmentRecord:
    try:
        data = path.read_bytes()
    except Exception as exc:
        raise _error("segment", sequence=sequence, detail="unable to read local segment file") from exc

    byte_length = len(data)
    sha256 = hashlib.sha256(data).hexdigest()

    return SegmentRecord(
        source_url=source_url,
        sequence=sequence,
        resolved_uri=resolved_uri,
        local_path=str(path),
        start_ts=float(start_ts),
        duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
        byte_length=byte_length,
        sha256=sha256,
        metadata=metadata,
        _loader=lambda segment_path=path: segment_path.read_bytes(),
    )


def _parse_media_sequence(line: str) -> int:
    value = line.removeprefix("#EXT-X-MEDIA-SEQUENCE:").strip()
    try:
        sequence = int(value)
    except ValueError as exc:
        raise _error("manifest", detail="malformed media sequence") from exc
    if sequence < 0:
        raise _error("manifest", detail="media sequence must be >= 0")
    return sequence


def _parse_extinf_duration(line: str, *, sequence: int) -> float:
    value = line.removeprefix("#EXTINF:").split(",", 1)[0].strip()
    try:
        duration = float(value)
    except ValueError as exc:
        raise _error("manifest", sequence=sequence, detail="malformed EXTINF duration") from exc
    if duration < 0:
        raise _error("manifest", sequence=sequence, detail="EXTINF duration must be >= 0")
    return duration


def _looks_like_network_url(value: str | Path) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"}


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()
