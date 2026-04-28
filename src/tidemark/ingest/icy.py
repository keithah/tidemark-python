"""ICY/Icecast metadata helpers."""

from __future__ import annotations

import re

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
