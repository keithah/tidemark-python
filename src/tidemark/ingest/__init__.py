"""Ingest helpers for supported input surfaces."""

from tidemark.ingest.hls import (
    HlsScte35Tag,
    direct_cue_marker,
    iter_hls_manifest_id3_markers,
    iter_hls_manifest_scte35_markers,
    parse_hls_scte35_tag,
)
from tidemark.ingest.icy import (
    DEFAULT_META_INT,
    icy_marker_from_fields,
    icy_request_headers,
    iter_icy_markers,
    parse_icy_metadata,
    sanitize_icy_metadata,
)
from tidemark.ingest.mpegts import iter_mpegts_scte35_markers
from tidemark.ingest.segments import (
    SegmentIngestError,
    SegmentRecord,
    resolve_local_hls_segments,
    resolve_local_media_file,
    resolve_segments,
)
from tidemark.ingest.udp import (
    UDPAddressError,
    UDPDatagramReader,
    UDP_DATAGRAM_SIZE,
    iter_udp_scte35_markers,
    open_udp_socket,
    parse_udp_url,
)

__all__ = [
    "DEFAULT_META_INT",
    "HlsScte35Tag",
    "SegmentIngestError",
    "SegmentRecord",
    "UDPAddressError",
    "UDPDatagramReader",
    "UDP_DATAGRAM_SIZE",
    "direct_cue_marker",
    "icy_marker_from_fields",
    "icy_request_headers",
    "iter_hls_manifest_id3_markers",
    "iter_hls_manifest_scte35_markers",
    "iter_icy_markers",
    "iter_mpegts_scte35_markers",
    "iter_udp_scte35_markers",
    "open_udp_socket",
    "parse_hls_scte35_tag",
    "parse_icy_metadata",
    "parse_udp_url",
    "resolve_local_hls_segments",
    "resolve_local_media_file",
    "resolve_segments",
    "sanitize_icy_metadata",
]
