"""HLS SCTE-35 tag parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterator
from urllib.parse import urljoin

from tidemark.markers.id3 import decode_id3_markers_from_segment_bytes
from tidemark.markers.models import AdMarker
from tidemark.markers.scte35 import decode_scte35_marker, decode_scte35_markers_from_mpegts


@dataclass(frozen=True)
class HlsScte35Tag:
    """Parsed SCTE-35-related HLS manifest tag data."""

    tag: str
    payload: str | None
    attributes: dict[str, str]
    direct_fields: dict[str, str]


def parse_hls_scte35_tag(line: str) -> HlsScte35Tag | None:
    """Parse one supported HLS SCTE-35 manifest line.

    Unsupported lines and SCTE-35 tags without usable payload/direct cue data return
    ``None`` so callers can scan manifests without routing non-cues into binary decode.
    """
    manifest_line = line.strip()
    if not manifest_line:
        return None

    if manifest_line.startswith("#EXT-X-SCTE35:"):
        body = manifest_line.removeprefix("#EXT-X-SCTE35:").strip()
        if not body:
            return None
        attributes = _parse_hls_attributes(body) if body.startswith("CUE=") else {}
        payload = attributes.get("CUE") if attributes else body
        return _payload_tag("#EXT-X-SCTE35", payload, attributes)

    if manifest_line.startswith("#EXT-OATCLS-SCTE35:"):
        payload = manifest_line.removeprefix("#EXT-OATCLS-SCTE35:").strip()
        return _payload_tag("#EXT-OATCLS-SCTE35", payload, {})

    if manifest_line.startswith("#EXT-X-DATERANGE:"):
        attributes = _parse_hls_attributes(manifest_line.removeprefix("#EXT-X-DATERANGE:"))
        payload = _first_present(attributes, ("SCTE35-OUT", "SCTE35-IN"))
        return _payload_tag("#EXT-X-DATERANGE", payload, attributes)

    if manifest_line.startswith("#EXT-X-CUE-OUT-CONT:"):
        attributes = _parse_hls_attributes(manifest_line.removeprefix("#EXT-X-CUE-OUT-CONT:"))
        payload = attributes.get("SCTE35")
        return _payload_tag("#EXT-X-CUE-OUT-CONT", payload, attributes)

    if manifest_line == "#EXT-X-CUE-OUT":
        return HlsScte35Tag(
            tag="#EXT-X-CUE-OUT",
            payload=None,
            attributes={},
            direct_fields={},
        )

    if manifest_line.startswith("#EXT-X-CUE-OUT:"):
        body = manifest_line.removeprefix("#EXT-X-CUE-OUT:").strip()
        attributes = _parse_direct_cue_attributes(body)
        return HlsScte35Tag(
            tag="#EXT-X-CUE-OUT",
            payload=None,
            attributes=attributes,
            direct_fields=dict(attributes),
        )

    if manifest_line == "#EXT-X-CUE-IN":
        return HlsScte35Tag(
            tag="#EXT-X-CUE-IN",
            payload=None,
            attributes={},
            direct_fields={},
        )

    return None


def iter_hls_manifest_scte35_markers(
    manifest_text: str,
    *,
    segment_loader: Callable[[str], bytes] | None = None,
    manifest_url: str | None = None,
    timestamp: float = 0.0,
) -> Iterator[AdMarker]:
    """Yield SCTE-35 markers attached to media segments in a manifest.

    Manifest SCTE-35 tags are yielded before any SCTE-35 cues discovered inside
    the same media segment bytes. Decode/load failures are wrapped with segment
    context while preserving payload, byte, and source URL redaction.
    """
    current_sequence = 0
    pending_tags: list[HlsScte35Tag] = []

    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                current_sequence = _parse_media_sequence(line)
                continue

            parsed = parse_hls_scte35_tag(line)
            if parsed is not None:
                pending_tags.append(parsed)
            continue

        if pending_tags:
            for pending_tag in pending_tags:
                yield _marker_from_pending_tag(
                    pending_tag,
                    segment=current_sequence,
                    timestamp=timestamp,
                )
            pending_tags = []

        if segment_loader is not None:
            yield from _markers_from_segment_bytes(
                line,
                segment_loader=segment_loader,
                manifest_url=manifest_url,
                segment=current_sequence,
                timestamp=timestamp,
            )

        current_sequence += 1


def iter_hls_manifest_id3_markers(
    manifest_text: str,
    *,
    segment_loader: Callable[[str], bytes],
    manifest_url: str | None = None,
    timestamp: float = 0.0,
) -> Iterator[AdMarker]:
    """Yield ID3 markers decoded from every media segment URI in an HLS manifest.

    Segment loading and decode failures are wrapped with redacted phase and
    segment context only; raw bytes, URLs, and parsed private frame data are not
    included in wrapper messages.
    """
    current_sequence = 0

    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                current_sequence = _parse_media_sequence(line)
            continue

        yield from _id3_markers_from_segment_bytes(
            line,
            segment_loader=segment_loader,
            manifest_url=manifest_url,
            segment=current_sequence,
            timestamp=timestamp,
        )
        current_sequence += 1


def direct_cue_marker(
    tag: str,
    fields: dict[str, str],
    segment: int | None = None,
    timestamp: float = 0.0,
) -> AdMarker:
    """Build an ``AdMarker`` for direct HLS CUE-IN/OUT tags without binary decode."""
    return AdMarker(
        type="SCTE35",
        classification="UNKNOWN",
        source="hls_manifest",
        tag=tag,
        segment=segment,
        fields=dict(fields),
        timestamp=timestamp,
    )


def _marker_from_pending_tag(tag: HlsScte35Tag, *, segment: int, timestamp: float) -> AdMarker:
    if tag.payload is None:
        return direct_cue_marker(
            tag.tag,
            tag.direct_fields,
            segment=segment,
            timestamp=timestamp,
        )

    try:
        return decode_scte35_marker(
            tag.payload,
            source="hls_manifest",
            tag=tag.tag,
            segment=segment,
            timestamp=timestamp,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unable to decode hls_manifest SCTE-35 marker for tag {tag.tag} at segment {segment}"
        ) from exc


def _markers_from_segment_bytes(
    uri: str,
    *,
    segment_loader: Callable[[str], bytes],
    manifest_url: str | None,
    segment: int,
    timestamp: float,
) -> Iterator[AdMarker]:
    resolved_uri = urljoin(manifest_url, uri) if manifest_url is not None else uri
    try:
        data = segment_loader(resolved_uri)
    except Exception as exc:
        raise ValueError(f"Unable to load HLS segment bytes at segment {segment}") from exc

    if not isinstance(data, bytes):
        raise TypeError("HLS segment loader must return bytes")

    try:
        yield from decode_scte35_markers_from_mpegts(
            data,
            source="hls_segment",
            segment=segment,
            timestamp=timestamp,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to decode hls_segment SCTE-35 markers at segment {segment}") from exc


def _id3_markers_from_segment_bytes(
    uri: str,
    *,
    segment_loader: Callable[[str], bytes],
    manifest_url: str | None,
    segment: int,
    timestamp: float,
) -> Iterator[AdMarker]:
    resolved_uri = urljoin(manifest_url, uri) if manifest_url is not None else uri
    try:
        data = segment_loader(resolved_uri)
    except Exception as exc:
        raise ValueError(f"Unable to load HLS segment bytes at segment {segment}") from exc

    if not isinstance(data, bytes):
        raise TypeError("HLS segment loader must return bytes")

    try:
        yield from decode_id3_markers_from_segment_bytes(
            data,
            source="hls_segment",
            tag="ID3",
            segment=segment,
            timestamp=timestamp,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to decode hls_segment ID3 markers at segment {segment}") from exc


def _parse_media_sequence(line: str) -> int:
    value = line.removeprefix("#EXT-X-MEDIA-SEQUENCE:").strip()
    try:
        sequence = int(value)
    except ValueError:
        return 0
    return max(sequence, 0)


def _payload_tag(tag: str, payload: str | None, attributes: dict[str, str]) -> HlsScte35Tag | None:
    if payload is None or not payload.strip():
        return None
    return HlsScte35Tag(
        tag=tag,
        payload=payload.strip(),
        attributes=attributes,
        direct_fields={},
    )


def _first_present(attributes: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = attributes.get(key)
        if value is not None:
            return value
    return None


def _parse_direct_cue_attributes(body: str) -> dict[str, str]:
    if not body:
        return {}
    if "=" not in body:
        return {"DURATION": _unquote_hls_attribute_value(body.strip())}
    return _parse_hls_attributes(body)


def _parse_hls_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for part in _split_unquoted_commas(text):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        attributes[key] = _unquote_hls_attribute_value(value.strip())
    return attributes


def _split_unquoted_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    in_quotes = False
    escaped = False

    for index, character in enumerate(text):
        if character == "\\" and in_quotes and not escaped:
            escaped = True
            continue
        if character == '"' and not escaped:
            in_quotes = not in_quotes
        elif character == "," and not in_quotes:
            parts.append(text[start:index])
            start = index + 1
        escaped = False

    parts.append(text[start:])
    return parts


def _unquote_hls_attribute_value(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value
