"""ID3 marker decoding helpers for HLS segment bytes."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from mutagen.id3 import ID3

from tidemark.markers.models import AdMarker

_ID3_HEADER = b"ID3"
_ID3_HEADER_SIZE = 10
_ID3_FOOTER_FLAG = 0x10


def decode_id3_markers_from_segment_bytes(
    data: bytes,
    *,
    source: str = "hls_segment",
    tag: str = "ID3",
    segment: int | None = None,
    timestamp: float = 0.0,
) -> list[AdMarker]:
    """Decode complete ID3v2 tags embedded in segment bytes into markers.

    Error messages intentionally name only the scan/parse phase and optional segment
    number. They do not include input bytes, full-segment base64, URLs, or parser
    exception text because HLS segment bytes may contain customer-private payloads.
    """
    if not isinstance(data, bytes):
        raise TypeError(_error_message("ID3 scan", segment, "segment data must be bytes"))
    if not data:
        return []

    markers: list[AdMarker] = []
    for tag_bytes in _iter_complete_id3_tags(data, segment=segment):
        parsed = _parse_id3_tag(tag_bytes, segment=segment)
        markers.append(
            AdMarker(
                type="ID3",
                classification="UNKNOWN",
                source=source,
                tag=tag,
                segment=segment,
                raw_base64=base64.b64encode(tag_bytes).decode("ascii"),
                tags=_id3_tags(parsed),
                fields=_id3_fields(parsed),
                timestamp=timestamp,
            )
        )
    return markers


def _iter_complete_id3_tags(data: bytes, *, segment: int | None) -> list[bytes]:
    tags: list[bytes] = []
    offset = 0

    while True:
        header_offset = data.find(_ID3_HEADER, offset)
        if header_offset == -1:
            return tags

        header_end = header_offset + _ID3_HEADER_SIZE
        if header_end > len(data):
            raise ValueError(_error_message("ID3 scan", segment, "truncated ID3 header"))

        header = data[header_offset:header_end]
        payload_size = _parse_synchsafe_size(header[6:10], segment=segment)
        total_size = _ID3_HEADER_SIZE + payload_size
        if header[5] & _ID3_FOOTER_FLAG:
            total_size += _ID3_HEADER_SIZE

        tag_end = header_offset + total_size
        if tag_end > len(data):
            raise ValueError(_error_message("ID3 scan", segment, "truncated ID3 tag"))

        tags.append(data[header_offset:tag_end])
        offset = tag_end


def _parse_synchsafe_size(size_bytes: bytes, *, segment: int | None) -> int:
    if len(size_bytes) != 4 or any(value & 0x80 for value in size_bytes):
        raise ValueError(_error_message("ID3 scan", segment, "malformed synchsafe size"))

    size = 0
    for value in size_bytes:
        size = (size << 7) | value
    return size


def _parse_id3_tag(tag_bytes: bytes, *, segment: int | None) -> ID3:
    try:
        return ID3(BytesIO(tag_bytes))
    except Exception as exc:  # pragma: no cover - dependency exception type varies.
        raise ValueError(_error_message("ID3 parse", segment, "unable to parse ID3 tag")) from exc


def _id3_tags(parsed: ID3) -> dict[str, str]:
    """Build a Go-compatible frame-id → value string map from a parsed ID3 tag.

    Matches the Go Tags map[string]string format: TIT2/TIT3 → decoded text,
    TXXX → "desc:value", PRIV → "owner:hexdata", GEOB → "mime:fn:desc:hexdata".
    For duplicate frame IDs, last one wins (consistent with Go map behavior).
    """
    result: dict[str, str] = {}
    for key, frame in sorted(parsed.items(), key=_frame_sort_key):
        frame_id = str(getattr(frame, "FrameID", key.split(":", 1)[0]))
        if frame_id in ("TIT2", "TIT3"):
            text_list = getattr(frame, "text", [])
            result[frame_id] = " ".join(str(v) for v in text_list)
        elif frame_id == "TXXX":
            desc = str(getattr(frame, "desc", ""))
            text_list = getattr(frame, "text", [])
            value = " ".join(str(v) for v in text_list)
            result[frame_id] = f"{desc}:{value}" if desc else value
        elif frame_id == "PRIV":
            owner = str(getattr(frame, "owner", ""))
            data = getattr(frame, "data", b"")
            if not isinstance(data, bytes):
                data = bytes(data)
            result[frame_id] = f"{owner}:{data.hex()}"
        elif frame_id == "GEOB":
            mime = str(getattr(frame, "mime", ""))
            filename = str(getattr(frame, "filename", ""))
            desc = str(getattr(frame, "desc", ""))
            data = getattr(frame, "data", b"")
            if not isinstance(data, bytes):
                data = bytes(data)
            result[frame_id] = f"{mime}:{filename}:{desc}:{data.hex()}"
    return result


def _id3_fields(parsed: ID3) -> dict[str, Any]:
    frame_entries = [_frame_to_dict(frame) for _, frame in sorted(parsed.items(), key=_frame_sort_key)]
    frame_ids = sorted({entry["ID"] for entry in frame_entries})
    return {"FrameIDs": frame_ids, "Frames": frame_entries}


def _frame_sort_key(item: tuple[str, Any]) -> tuple[str, str]:
    key, frame = item
    frame_id = str(getattr(frame, "FrameID", key.split(":", 1)[0]))
    return frame_id, key


def _frame_to_dict(frame: Any) -> dict[str, Any]:
    frame_id = str(getattr(frame, "FrameID", type(frame).__name__))

    if frame_id == "PRIV":
        data = getattr(frame, "data", b"")
        if not isinstance(data, bytes):
            data = bytes(data)
        return {
            "ID": "PRIV",
            "Owner": str(getattr(frame, "owner", "")),
            "DataBase64": base64.b64encode(data).decode("ascii"),
            "DataLength": len(data),
        }

    if frame_id == "TXXX":
        return {
            "ID": "TXXX",
            "Description": str(getattr(frame, "desc", "")),
            "Text": _text_values(frame),
        }

    if hasattr(frame, "text"):
        return {"ID": frame_id, "Text": _text_values(frame)}

    return {"ID": frame_id, "FrameType": type(frame).__name__}


def _text_values(frame: Any) -> list[str]:
    text = getattr(frame, "text", [])
    if isinstance(text, (str, bytes)):
        text = [text]
    return [value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value) for value in text]


def _error_message(phase: str, segment: int | None, detail: str) -> str:
    location = f" for segment {segment}" if segment is not None else ""
    return f"{phase} failed{location}: {detail}"
