"""Local media segment resolution helpers.

This module turns deterministic local fixture inputs into typed segment records for
later persistence/transcription handoff. It deliberately does not decode markers,
call monitor source adapters, or fetch network URLs.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urljoin, urlparse


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


def iter_live_hls_segments(
    source: str,
    *,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> Iterator[SegmentRecord]:
    """Yield newly discovered segments from a live network HLS playlist until timeout.

    This is intentionally network-only and bounded by *timeout* when provided. VOD
    playlists with ``#EXT-X-ENDLIST`` are yielded once and then stop.
    """
    if not _looks_like_network_url(source):
        raise _error("source", detail="live HLS ingest requires a network URL")

    from tidemark.monitor_sources import _load_hls_manifest_text, _resolve_hls_manifest

    deadline = None if timeout is None else time.monotonic() + timeout
    raw_text = _load_hls_manifest_text(source, timeout=timeout, headers=headers)
    manifest_text, media_url = _resolve_hls_manifest(raw_text, source=source, timeout=timeout, headers=headers)
    manifest_url = media_url or source
    seen: set[str] = set()
    emitted_any = False

    while True:
        for segment in _network_hls_segments_from_manifest(
            manifest_text,
            manifest_url=manifest_url,
            seen=seen,
            timeout=timeout,
            headers=headers,
        ):
            emitted_any = True
            yield segment

        if _hls_has_endlist(manifest_text):
            return
        if _deadline_reached(deadline):
            return
        time.sleep(_poll_delay(manifest_text, deadline=deadline))
        if _deadline_reached(deadline):
            return
        manifest_text = _load_hls_manifest_text(manifest_url, timeout=timeout, headers=headers)
        if not emitted_any and _deadline_reached(deadline):
            return


def _network_hls_segments_from_manifest(
    manifest_text: str,
    *,
    manifest_url: str,
    seen: set[str],
    timeout: float | None,
    headers: dict[str, str] | None,
) -> Iterator[SegmentRecord]:
    from tidemark.monitor_sources import _open_http_response

    sequence = 0
    start_ts = 0.0
    pending_duration: float | None = None

    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                sequence = _parse_live_media_sequence(line)
                continue
            if line.startswith("#EXTINF:"):
                pending_duration = _parse_extinf_duration(line, sequence=sequence)
                continue
            continue

        duration = pending_duration
        pending_duration = None
        resolved_uri = urljoin(manifest_url, line)
        if resolved_uri in seen:
            start_ts += duration or 0.0
            sequence += 1
            continue

        try:
            with _open_http_response(resolved_uri, timeout=timeout, headers=headers) as response:
                data = response.read()
        except Exception as exc:
            raise _error("segment", sequence=sequence, detail="unable to read network segment") from exc
        if not isinstance(data, bytes):
            raise _error("segment", sequence=sequence, detail="network segment response must be bytes")

        seen.add(resolved_uri)
        byte_length = len(data)
        sha256 = hashlib.sha256(data).hexdigest()
        yield SegmentRecord(
            source_url=manifest_url,
            sequence=sequence,
            resolved_uri=resolved_uri,
            local_path=None,
            start_ts=start_ts,
            duration_seconds=duration,
            byte_length=byte_length,
            sha256=sha256,
            metadata={"source_label": "live_hls"},
            _loader=lambda payload=data: payload,
        )
        start_ts += duration or 0.0
        sequence += 1


def _hls_has_endlist(manifest_text: str) -> bool:
    return any(line.strip() == "#EXT-X-ENDLIST" for line in manifest_text.splitlines())


def _poll_delay(manifest_text: str, *, deadline: float | None) -> float:
    target = 1.0
    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("#EXT-X-TARGETDURATION:"):
            continue
        try:
            target = max(float(line.removeprefix("#EXT-X-TARGETDURATION:").strip()), 0.25)
        except ValueError:
            target = 1.0
        break
    delay = min(max(target / 2.0, 0.25), 2.0)
    if deadline is None:
        return delay
    return max(0.0, min(delay, deadline - time.monotonic()))


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _parse_live_media_sequence(line: str) -> int:
    try:
        return max(int(line.removeprefix("#EXT-X-MEDIA-SEQUENCE:").strip()), 0)
    except ValueError as exc:
        raise _error("manifest", detail="malformed media sequence") from exc


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
