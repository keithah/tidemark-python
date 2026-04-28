"""SCTE-35 decoding helpers for tidemark-owned marker models."""

from __future__ import annotations

import base64
import io
from typing import Any

import threefive

from tidemark.markers.models import AdMarker


def decode_scte35_marker(
    payload: str | bytes,
    source: str,
    tag: str | None = None,
    segment: int | None = None,
    timestamp: float = 0.0,
) -> AdMarker:
    """Decode one SCTE-35 payload into the Go-compatible ``AdMarker`` contract.

    Decode failures deliberately omit the input payload from error messages because
    future call sites may pass source manifests or private SCTE-35 data.
    """
    cue_text, raw_base64 = _normalize_payload_text(payload)

    try:
        cue = threefive.Cue(cue_text)
    except Exception as exc:  # pragma: no cover - exact dependency exception varies.
        raise ValueError("Unable to initialize SCTE-35 cue") from exc

    try:
        decoded = cue.decode()
    except Exception as exc:
        raise ValueError("Unable to decode SCTE-35 cue") from exc

    if decoded is not True:
        raise ValueError("Unable to decode SCTE-35 cue")

    return _cue_to_ad_marker(
        cue,
        source=source,
        tag=tag,
        segment=segment,
        timestamp=timestamp,
        raw_base64=raw_base64,
    )


def decode_scte35_markers_from_mpegts(
    data: bytes,
    *,
    source: str = "hls_segment",
    tag: str | None = None,
    segment: int | None = None,
    timestamp: float = 0.0,
) -> list[AdMarker]:
    """Decode SCTE-35 cues from MPEGTS segment bytes into ``AdMarker`` records.

    Dependency failures deliberately omit byte contents because callers may pass
    private transport stream segments or URL-loaded customer media.
    """
    if not isinstance(data, bytes):
        raise TypeError("SCTE-35 MPEGTS segment data must be bytes")
    if not data:
        return []

    cues: list[threefive.Cue] = []

    def collect_cue(cue: threefive.Cue) -> None:
        cues.append(cue)

    try:
        stream = threefive.Stream(io.BytesIO(data))
        stream.decode(collect_cue)
    except Exception as exc:
        raise ValueError("Unable to decode SCTE-35 markers from MPEGTS segment bytes") from exc

    markers: list[AdMarker] = []
    for cue in cues:
        try:
            raw_base64 = cue.encode()
        except Exception:
            raw_base64 = None
        markers.append(
            _cue_to_ad_marker(
                cue,
                source=source,
                tag=tag,
                segment=segment,
                timestamp=timestamp,
                raw_base64=raw_base64,
            )
        )
    return markers


def _normalize_payload_text(payload: str | bytes) -> tuple[str, str]:
    if isinstance(payload, bytes):
        try:
            payload_text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Unable to decode SCTE-35 payload bytes") from exc
    else:
        try:
            payload.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Unable to decode SCTE-35 payload bytes") from exc
        payload_text = payload

    payload_text = payload_text.strip()
    if not payload_text:
        raise ValueError("Unable to decode SCTE-35 payload: empty payload")

    if payload_text.lower().startswith("0x"):
        hex_text = payload_text[2:]
        if not hex_text:
            raise ValueError("Unable to decode SCTE-35 payload bytes")
        try:
            payload_bytes = bytes.fromhex(hex_text)
        except ValueError as exc:
            raise ValueError("Unable to decode SCTE-35 payload bytes") from exc
        raw_base64 = base64.b64encode(payload_bytes).decode("ascii")
        return raw_base64, raw_base64

    return payload_text, payload_text


def _cue_to_ad_marker(
    cue: threefive.Cue,
    *,
    source: str,
    tag: str | None,
    segment: int | None,
    timestamp: float,
    raw_base64: str | None,
) -> AdMarker:
    command = _get_command(cue)
    descriptors = _get_descriptors(cue)
    fields = _build_fields(command, descriptors)

    return AdMarker(
        type="SCTE35",
        classification="UNKNOWN",
        source=source,
        tag=tag,
        pts=_number_or_none(command.get("pts_time")),
        segment=segment,
        break_duration=_number_or_none(command.get("break_duration")),
        raw_base64=raw_base64,
        command=command,
        descriptors=descriptors,
        fields=fields,
        timestamp=timestamp,
    )


def _get_command(cue: threefive.Cue) -> dict[str, Any]:
    if cue.command is None or not hasattr(cue.command, "get"):
        raise ValueError("Decoded SCTE-35 cue is missing command data")

    try:
        command = cue.command.get()
    except Exception as exc:
        raise ValueError("Unable to normalize SCTE-35 command data") from exc

    if not isinstance(command, dict) or not command.get("name"):
        raise ValueError("Decoded SCTE-35 cue has malformed command data")

    return dict(command)


def _get_descriptors(cue: threefive.Cue) -> list[dict[str, Any]]:
    try:
        descriptors = cue.get_descriptors()
    except Exception as exc:
        raise ValueError("Unable to normalize SCTE-35 descriptor data") from exc

    if descriptors is None:
        return []
    if not isinstance(descriptors, list):
        raise ValueError("Decoded SCTE-35 cue has malformed descriptor data")

    normalized: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("Decoded SCTE-35 cue has malformed descriptor data")
        normalized.append(dict(descriptor))
    return normalized


def _build_fields(command: dict[str, Any], descriptors: list[dict[str, Any]]) -> dict[str, str]:
    fields: dict[str, str] = {"CommandName": str(command["name"])}

    out_of_network = command.get("out_of_network_indicator")
    if isinstance(out_of_network, bool):
        fields["OutOfNetworkIndicator"] = "true" if out_of_network else "false"

    break_duration = _number_or_none(command.get("break_duration"))
    if break_duration is not None:
        fields["BreakDuration"] = f"{break_duration:.3f}"

    splice_event_id = command.get("splice_event_id")
    if isinstance(splice_event_id, int):
        fields["SpliceEventID"] = f"0x{splice_event_id:x}"

    for descriptor in descriptors:
        segmentation_type_id = descriptor.get("segmentation_type_id")
        if isinstance(segmentation_type_id, int):
            fields["SegmentationTypeID"] = f"0x{segmentation_type_id:02x}"
            break

    return fields


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
