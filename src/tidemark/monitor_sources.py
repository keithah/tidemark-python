"""Importable source detection and marker adapters for monitor runtime.

This module stays CLI-free so Typer/console parsing can layer on top later. It owns
only conservative stream-type routing and redacted wrapping around source setup or
iteration failures.
"""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Callable, Iterator
from enum import Enum
from os import PathLike
from pathlib import Path
from urllib.parse import urlparse

from tidemark.ingest.mpegts import iter_mpegts_scte35_markers
from tidemark.ingest.udp import UDPAddressError, iter_udp_scte35_markers
from tidemark.markers import AdMarker

SourceInput = str | PathLike[str]


class StreamType(str, Enum):
    """Supported monitor source types."""

    AUTO = "auto"
    MPEGTS = "mpegts"
    UDP = "udp"
    HLS = "hls"
    ICY = "icy"


class MonitorSourceError(ValueError):
    """Raised for redacted monitor source routing/setup/iteration failures."""

    def __init__(self, message: str, *, stream_type: StreamType | None = None, phase: str | None = None):
        super().__init__(message)
        self.stream_type = stream_type
        self.phase = phase


_STREAM_TYPE_ALIASES = {
    "auto": StreamType.AUTO,
    "mpegts": StreamType.MPEGTS,
    "mpeg-ts": StreamType.MPEGTS,
    "ts": StreamType.MPEGTS,
    "udp": StreamType.UDP,
    "hls": StreamType.HLS,
    "icy": StreamType.ICY,
}
_NETWORK_SCHEMES = {"http", "https"}
_PLAYLIST_SUFFIX = ".m3u8"
_DEFAULT_UDP_TIMEOUT_SECONDS = 2.0


def normalize_stream_type(requested: str | StreamType | None = StreamType.AUTO) -> StreamType:
    """Normalize a requested stream type or raise a redacted source error."""
    if requested is None:
        return StreamType.AUTO
    if isinstance(requested, StreamType):
        return requested

    normalized = _STREAM_TYPE_ALIASES.get(str(requested).strip().lower())
    if normalized is None:
        raise MonitorSourceError(
            f"invalid stream type: {requested}",
            phase="detection",
        )
    return normalized


def detect_stream_type(source: SourceInput, *, requested: str | StreamType | None = StreamType.AUTO) -> StreamType:
    """Detect a source stream type using conservative Go-compatible monitor routing.

    Explicit requests always win after validation. Auto-detection recognizes UDP URL
    forms, unambiguous bare IP UDP hostports, local/HTTP playlists as HLS, existing
    local non-playlist files as MPEGTS, and HTTP(S) non-playlists as MPEGTS.
    """
    requested_type = normalize_stream_type(requested)
    if requested_type is not StreamType.AUTO:
        return requested_type

    source_text = str(source).strip()
    parsed = urlparse(source_text)
    scheme = parsed.scheme.lower()

    if source_text.startswith("udp://"):
        return StreamType.UDP

    if _is_unambiguous_bare_udp(source_text):
        return StreamType.UDP

    if scheme in _NETWORK_SCHEMES:
        return StreamType.HLS if _is_playlist_path(parsed.path) else StreamType.MPEGTS

    path = Path(source)
    if path.exists():
        return StreamType.HLS if _is_playlist_path(path.name) else StreamType.MPEGTS

    raise MonitorSourceError("source setup failed: unable to detect stream type", phase="setup")


def iter_markers_for_source(
    source: SourceInput,
    *,
    stream_type: str | StreamType | None = StreamType.AUTO,
    timestamp_fn: Callable[[], float] = time.time,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> Iterator[AdMarker]:
    """Yield markers from ``source`` using the detected or explicit adapter.

    The returned iterator wraps adapter setup/iteration failures in stable,
    source-redacted :class:`MonitorSourceError` messages. Raw URLs, query strings,
    headers, packets, and payload bytes are intentionally not included.
    """
    detected_type = detect_stream_type(source, requested=stream_type)

    if detected_type is StreamType.MPEGTS:
        try:
            yield from iter_mpegts_scte35_markers(
                source,
                timestamp_fn=timestamp_fn,
                show_null=True,
                headers=headers,
            )
        except Exception as exc:
            raise MonitorSourceError(
                "mpegts source iteration failed",
                stream_type=detected_type,
                phase="iteration",
            ) from exc
        return

    if detected_type is StreamType.UDP:
        try:
            yield from iter_udp_scte35_markers(
                str(source),
                timestamp_fn=timestamp_fn,
                show_null=True,
                timeout=_DEFAULT_UDP_TIMEOUT_SECONDS if timeout is None else timeout,
            )
        except UDPAddressError as exc:
            raise MonitorSourceError(
                "udp source setup failed",
                stream_type=detected_type,
                phase="setup",
            ) from exc
        except Exception as exc:
            raise MonitorSourceError(
                "udp source iteration failed",
                stream_type=detected_type,
                phase="iteration",
            ) from exc
        return

    raise MonitorSourceError(
        f"{detected_type.value} source adapter not implemented",
        stream_type=detected_type,
        phase="setup",
    )


# Compatibility name for downstream tasks/planned callers.
monitor_source = iter_markers_for_source


def _is_playlist_path(path: str) -> bool:
    return path.lower().split("?", 1)[0].endswith(_PLAYLIST_SUFFIX)


def _is_unambiguous_bare_udp(source_text: str) -> bool:
    candidate = source_text.strip()
    if not candidate or "://" in candidate:
        return False

    used_at = candidate.startswith("@")
    if used_at:
        candidate = candidate[1:]

    host, separator, port_text = candidate.rpartition(":")
    if not separator or not host or not port_text.isdigit():
        return False

    try:
        port = int(port_text)
    except ValueError:
        return False
    if not 0 < port <= 65535:
        return False

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


__all__ = [
    "MonitorSourceError",
    "SourceInput",
    "StreamType",
    "detect_stream_type",
    "iter_markers_for_source",
    "monitor_source",
    "normalize_stream_type",
]
