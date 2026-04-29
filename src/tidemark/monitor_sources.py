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
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from tidemark.ingest.hls import iter_hls_manifest_id3_markers, iter_hls_manifest_scte35_markers
from tidemark.ingest.icy import DEFAULT_META_INT, icy_request_headers, iter_icy_markers
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
    "icecast": StreamType.ICY,
}
_NETWORK_SCHEMES = {"http", "https"}
_PLAYLIST_SUFFIX = ".m3u8"
_DEFAULT_UDP_TIMEOUT_SECONDS = 2.0
_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0


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
    requested_type = normalize_stream_type(stream_type)
    detected_type = detect_stream_type(source, requested=requested_type)
    sniffed_response: object | None = None
    sniffed_manifest_text: str | None = None

    if requested_type is StreamType.AUTO and detected_type is StreamType.MPEGTS and _is_network_source(source):
        sniffed_type, sniffed_response, sniffed_manifest_text = _sniff_http_source(
            str(source),
            timeout=timeout,
            headers=headers,
        )
        if sniffed_type is not None:
            detected_type = sniffed_type

    if detected_type is StreamType.MPEGTS:
        _close_response(sniffed_response)
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

    if detected_type is StreamType.HLS:
        yield from _iter_hls_source(
            source,
            timestamp_fn=timestamp_fn,
            timeout=timeout,
            headers=headers,
            manifest_text=sniffed_manifest_text,
            response=sniffed_response,
        )
        return

    if detected_type is StreamType.ICY:
        yield from _iter_icy_source(
            source,
            timestamp_fn=timestamp_fn,
            timeout=timeout,
            headers=headers,
            response=sniffed_response,
        )
        return

    raise MonitorSourceError(
        f"{detected_type.value} source adapter not implemented",
        stream_type=detected_type,
        phase="setup",
    )


# Compatibility name for downstream tasks/planned callers.
monitor_source = iter_markers_for_source


def _iter_hls_source(
    source: SourceInput,
    *,
    timestamp_fn: Callable[[], float],
    timeout: float | None,
    headers: dict[str, str] | None,
    manifest_text: str | None = None,
    response: object | None = None,
) -> Iterator[AdMarker]:
    try:
        active_manifest_text = manifest_text
        effective_source: SourceInput = source
        if active_manifest_text is None:
            raw_text = _load_hls_manifest_text(source, timeout=timeout, headers=headers, response=response)
            active_manifest_text, media_url = _resolve_hls_manifest(
                raw_text, source=source, timeout=timeout, headers=headers
            )
            if media_url is not None:
                effective_source = media_url
        segment_loader = _build_hls_segment_loader(effective_source, timeout=timeout, headers=headers)
    except Exception as exc:
        _close_response(response)
        raise MonitorSourceError("hls source setup failed", stream_type=StreamType.HLS, phase="setup") from exc

    seen: set[tuple[object, ...]] = set()
    try:
        timestamp = timestamp_fn()
        for marker_iterator_factory in (
            iter_hls_manifest_scte35_markers,
            iter_hls_manifest_id3_markers,
        ):
            marker_iterator = marker_iterator_factory(
                active_manifest_text,
                segment_loader=segment_loader,
                manifest_url=str(effective_source),
                timestamp=timestamp,
            )
            for marker in marker_iterator:
                key = _hls_marker_key(marker)
                if key in seen:
                    continue
                seen.add(key)
                yield marker
    except Exception as exc:
        raise MonitorSourceError("hls source iteration failed", stream_type=StreamType.HLS, phase="iteration") from exc


def _resolve_hls_manifest(
    manifest_text: str,
    *,
    source: SourceInput,
    timeout: float | None,
    headers: dict[str, str] | None,
) -> tuple[str, str | None]:
    """If *manifest_text* is an HLS master playlist, fetch and return the first variant.

    Returns ``(media_manifest_text, media_url)`` where *media_url* is the absolute URL
    of the resolved media playlist (used as the effective source for segment loading).
    Returns ``(manifest_text, None)`` unchanged when no master variant is found.
    """
    if "#EXT-X-STREAM-INF" not in manifest_text:
        return manifest_text, None

    source_str = str(source)
    for line in manifest_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        media_url = urljoin(source_str, line) if _is_network_source(source) else line
        media_text = _load_hls_manifest_text(media_url, timeout=timeout, headers=headers)
        return media_text, media_url

    return manifest_text, None


def _load_hls_manifest_text(
    source: SourceInput,
    *,
    timeout: float | None,
    headers: dict[str, str] | None,
    response: object | None = None,
) -> str:
    if response is not None:
        body = response.read()
        _close_response(response)
        return _decode_manifest_body(body)

    if _is_network_source(source):
        with _open_http_response(str(source), timeout=timeout, headers=headers) as http_response:
            return _decode_manifest_body(http_response.read())

    return Path(source).read_text(encoding="utf-8")


def _decode_manifest_body(body: object) -> str:
    if not isinstance(body, bytes):
        raise TypeError("HLS manifest response body must be bytes")
    return body.decode("utf-8")


def _build_hls_segment_loader(
    source: SourceInput,
    *,
    timeout: float | None,
    headers: dict[str, str] | None,
) -> Callable[[str], bytes]:
    manifest_text = str(source)
    parsed_manifest = urlparse(manifest_text)

    if parsed_manifest.scheme.lower() in _NETWORK_SCHEMES:

        def load_network_segment(uri: str) -> bytes:
            resolved_uri = urljoin(manifest_text, uri)
            with _open_http_response(resolved_uri, timeout=timeout, headers=headers) as response:
                body = response.read()
            if not isinstance(body, bytes):
                raise TypeError("HLS segment loader must return bytes")
            return body

        return load_network_segment

    manifest_dir = Path(source).parent

    def load_local_segment(uri: str) -> bytes:
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme.lower() in _NETWORK_SCHEMES:
            with _open_http_response(uri, timeout=timeout, headers=headers) as response:
                body = response.read()
            if not isinstance(body, bytes):
                raise TypeError("HLS segment loader must return bytes")
            return body
        return (manifest_dir / uri).read_bytes()

    return load_local_segment


def _hls_marker_key(marker: AdMarker) -> tuple[object, ...]:
    fields_items = _freeze_for_key(marker.fields or {})
    return (
        marker.type,
        marker.source,
        marker.tag,
        marker.segment,
        marker.raw_base64,
        _freeze_for_key(marker.command),
        fields_items,
    )


def _freeze_for_key(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_for_key(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_for_key(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_for_key(item) for item in value)
    return value


def _iter_icy_source(
    source: SourceInput,
    *,
    timestamp_fn: Callable[[], float],
    timeout: float | None,
    headers: dict[str, str] | None,
    response: object | None = None,
) -> Iterator[AdMarker]:
    active_response = response
    try:
        if active_response is None:
            active_response = _open_http_response(
                str(source),
                timeout=timeout,
                headers={**icy_request_headers(), **(headers or {})},
            )
        meta_int = _icy_meta_int(active_response)
    except Exception as exc:
        _close_response(active_response)
        raise MonitorSourceError("icy source setup failed", stream_type=StreamType.ICY, phase="setup") from exc

    try:
        yield from iter_icy_markers(active_response, meta_int=meta_int, source="icy_stream", timestamp=timestamp_fn)
    except Exception as exc:
        raise MonitorSourceError("icy source iteration failed", stream_type=StreamType.ICY, phase="iteration") from exc
    finally:
        _close_response(active_response)


def _sniff_http_source(
    source: str,
    *,
    timeout: float | None,
    headers: dict[str, str] | None,
) -> tuple[StreamType | None, object | None, str | None]:
    try:
        response = _open_http_response(
            source,
            timeout=timeout,
            headers={**icy_request_headers(), **(headers or {})},
        )
    except Exception:
        return None, None, None

    if _get_header(response, "icy-metaint") is not None:
        return StreamType.ICY, response, None

    try:
        body = response.read()
    except Exception:
        _close_response(response)
        return None, None, None

    if isinstance(body, bytes) and body.lstrip().startswith(b"#EXTM3U"):
        _close_response(response)
        return StreamType.HLS, None, _decode_manifest_body(body)

    _close_response(response)
    return None, None, None


def _open_http_response(url: str, *, timeout: float | None, headers: dict[str, str] | None):
    request = Request(url, headers=headers or {})
    return urlopen(request, timeout=_http_timeout(timeout))


def _http_timeout(timeout: float | None) -> float:
    return _DEFAULT_HTTP_TIMEOUT_SECONDS if timeout is None else timeout


def _icy_meta_int(response: object) -> int:
    raw_value = _get_header(response, "icy-metaint")
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_META_INT
    try:
        meta_int = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError("invalid ICY metadata interval") from exc
    if meta_int <= 0:
        raise ValueError("invalid ICY metadata interval")
    return meta_int


def _get_header(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value)

    lower_name = name.lower()
    items = getattr(headers, "items", None)
    if callable(items):
        for key, value in items():
            if str(key).lower() == lower_name:
                return str(value)
    return None


def _close_response(response: object | None) -> None:
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _is_playlist_path(path: str) -> bool:
    return path.lower().split("?", 1)[0].endswith(_PLAYLIST_SUFFIX)


def _is_network_source(source: SourceInput) -> bool:
    return urlparse(str(source)).scheme.lower() in _NETWORK_SCHEMES


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
