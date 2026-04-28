"""MPEGTS ingest helpers for decoded SCTE-35 marker streams."""

from __future__ import annotations

import time
from os import PathLike
from typing import Callable, Iterator

import threefive

from tidemark.markers.models import AdMarker
from tidemark.markers.scte35 import marker_from_scte35_cue

SourceInput = str | PathLike[str]

_MPEGTS_DECODE_ERROR = "Unable to decode MPEGTS SCTE-35 markers"


def iter_mpegts_scte35_markers(
    source: SourceInput,
    *,
    timestamp_fn: Callable[[], float] = time.time,
    show_null: bool = True,
    headers: dict[str, str] | None = None,
) -> Iterator[AdMarker]:
    """Yield Go-compatible SCTE-35 markers from a local or HTTP MPEGTS source.

    ``threefive.Stream`` owns file/HTTP opening and stream iteration. This wrapper
    only normalizes decoded cues into tidemark markers and redacts any source URL
    or private payload detail from raised error text.
    """
    stream_headers = {} if headers is None else headers

    try:
        stream = threefive.Stream(str(source), show_null=show_null, headers=stream_headers)
        for cue in stream.decode_next():
            yield marker_from_scte35_cue(
                cue,
                source="mpegts",
                timestamp=timestamp_fn(),
                raw_base64=_raw_base64_from_cue(cue),
            )
    except Exception as exc:
        raise ValueError(_MPEGTS_DECODE_ERROR) from exc


def _raw_base64_from_cue(cue: threefive.Cue) -> str | None:
    try:
        return cue.encode()
    except Exception:
        return None
