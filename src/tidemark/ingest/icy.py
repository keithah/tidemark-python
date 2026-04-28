"""ICY/Icecast metadata helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import BinaryIO

from tidemark.markers import AdMarker

DEFAULT_META_INT = 16000
_BINARY_PLACEHOLDER = "[binary data]"
_FIELD_PATTERN = re.compile(r"([^=;\s]+)='([^']*)';?")


def icy_request_headers() -> dict[str, str]:
    """Return headers that ask Icecast/Shoutcast servers to include ICY metadata."""
    return {"Icy-MetaData": "1"}


def sanitize_icy_metadata(meta: bytes) -> str:
    """Decode one ICY metadata block after removing null padding.

    Invalid UTF-8 is intentionally collapsed to a fixed placeholder so callers can
    handle malformed metadata without leaking raw stream bytes or station data.
    """
    if not isinstance(meta, bytes):
        raise TypeError("ICY metadata block must be bytes")

    stripped = meta.rstrip(b"\x00")
    if not stripped:
        return ""

    try:
        return stripped.decode("utf-8")
    except UnicodeDecodeError:
        return _BINARY_PLACEHOLDER


def parse_icy_metadata(meta: bytes) -> dict[str, str]:
    """Parse semicolon-delimited ``key='value'`` pairs from an ICY metadata block."""
    sanitized = sanitize_icy_metadata(meta)
    if not sanitized or sanitized == _BINARY_PLACEHOLDER:
        return {}

    return {match.group(1): match.group(2) for match in _FIELD_PATTERN.finditer(sanitized)}


def icy_marker_from_fields(
    fields: dict[str, str],
    source: str = "icy_stream",
    timestamp: float = 0.0,
) -> AdMarker | None:
    """Create an ICY ``AdMarker`` for non-empty ``StreamTitle`` fields."""
    title = fields.get("StreamTitle", "")
    if not title:
        return None

    return AdMarker(
        type="ICY",
        classification="UNKNOWN",
        source=source,
        fields=dict(fields),
        timestamp=timestamp,
    )


def _read_stream_bytes(stream: BinaryIO, size: int, phase: str) -> bytes:
    chunk = stream.read(size)
    if not isinstance(chunk, bytes):
        raise TypeError(f"ICY stream {phase} read must return bytes")
    return chunk


def _timestamp_value(timestamp: float | Callable[[], float]) -> float:
    if callable(timestamp):
        return float(timestamp())
    return float(timestamp)


def iter_icy_markers(
    stream: BinaryIO,
    meta_int: int = DEFAULT_META_INT,
    source: str = "icy_stream",
    timestamp: float | Callable[[], float] = 0.0,
) -> Iterator[AdMarker]:
    """Yield unique-title markers from an ICY audio stream with interleaved metadata.

    The iterator consumes one ICY frame at a time: ``meta_int`` audio bytes, one
    metadata length byte, then ``length * 16`` metadata bytes. Malformed framing
    errors intentionally include only phase context and byte counts, never raw
    stream data or station metadata.
    """
    if meta_int <= 0:
        raise ValueError("ICY metadata interval must be greater than zero")

    last_title: str | None = None

    while True:
        audio = _read_stream_bytes(stream, meta_int, "audio")
        if not audio:
            return
        if len(audio) < meta_int:
            raise ValueError("ICY metadata interval ended before a complete audio frame")

        length_byte = _read_stream_bytes(stream, 1, "metadata length")
        if not length_byte:
            raise ValueError("ICY metadata block length missing after audio frame")
        if len(length_byte) != 1:
            raise ValueError("ICY metadata block length read was incomplete")

        metadata_length = length_byte[0] * 16
        if metadata_length == 0:
            continue

        metadata = _read_stream_bytes(stream, metadata_length, "metadata block")
        if len(metadata) != metadata_length:
            raise ValueError("ICY metadata block ended before declared length")

        fields = parse_icy_metadata(metadata)
        title = fields.get("StreamTitle", "")
        if not title or title == last_title:
            continue

        marker = icy_marker_from_fields(fields, source=source, timestamp=_timestamp_value(timestamp))
        if marker is None:
            continue

        last_title = title
        yield marker
