"""SCTE-35 decoding helpers for tidemark-owned marker models."""

from __future__ import annotations

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
    raw_base64 = _payload_to_text(payload)
    if not raw_base64:
        raise ValueError("Unable to decode SCTE-35 payload: empty payload")

    try:
        cue = threefive.Cue(payload)
    except Exception as exc:  # pragma: no cover - exact dependency exception varies.
        raise ValueError("Unable to initialize SCTE-35 cue") from exc

    try:
        decoded = cue.decode()
    except Exception as exc:
        raise ValueError("Unable to decode SCTE-35 cue") from exc

    if decoded is not True:
        raise ValueError("Unable to decode SCTE-35 cue")

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


def _payload_to_text(payload: str | bytes) -> str:
    if isinstance(payload, bytes):
        try:
            return payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Unable to decode SCTE-35 payload bytes") from exc
    return payload


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
