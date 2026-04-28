"""Go-parity marker classification rules.

The classifier is intentionally a pure marker-layer component: it performs no I/O,
imports no decoders, and exposes classification strings plus marker mutations as its
only diagnostic surface.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from tidemark.markers.models import AdMarker

UNKNOWN = "UNKNOWN"
AD_START = "AD_START"
AD_END = "AD_END"

_SCTE35_START_SEGMENTATION_IDS = {0x22, 0x30, 0x34}
_SCTE35_END_SEGMENTATION_IDS = {0x23, 0x31, 0x35}
_ID3_END_SUBSTRINGS = ("ad_end", "content_start")
_ID3_START_WORDS = {"ad", "spot", "promo", "commercial"}
_WORD_RE = re.compile(r"\w+")


class Classifier:
    """Stateful marker classifier.

    One instance should be used per monitored stream so ICY ad-state transitions do
    not bleed between unrelated streams.
    """

    def __init__(self) -> None:
        self._icy_in_ad = False

    def classify(self, marker: AdMarker) -> str:
        """Classify ``marker``, mutate ``marker.classification``, and return it."""
        classification = self._classify(marker)
        marker.classification = classification
        return classification

    def _classify(self, marker: AdMarker) -> str:
        hls_classification = _classify_hls_tag(marker)
        if hls_classification != UNKNOWN:
            return hls_classification

        marker_type = _lower_text(getattr(marker, "type", ""))
        if marker_type == "scte35":
            return _classify_scte35(getattr(marker, "fields", None))
        if marker_type == "id3":
            return _classify_id3(marker)
        if marker_type == "icy":
            return self._classify_icy(getattr(marker, "fields", None))
        return UNKNOWN

    def _classify_icy(self, fields: object) -> str:
        if not isinstance(fields, dict):
            return UNKNOWN

        title = fields.get("StreamTitle")
        if not isinstance(title, str):
            return UNKNOWN

        title_classification = _classify_id3_candidate(title)
        if title_classification == AD_START:
            if self._icy_in_ad:
                return UNKNOWN
            self._icy_in_ad = True
            return AD_START

        if self._icy_in_ad:
            self._icy_in_ad = False
            return AD_END

        return UNKNOWN


def classify_marker(marker: AdMarker) -> str:
    """Classify a single marker with a short-lived classifier."""
    return Classifier().classify(marker)


def classify_markers(markers: Iterable[AdMarker], classifier: Classifier | None = None) -> list[str]:
    """Classify markers in order and return their classification strings."""
    active_classifier = classifier or Classifier()
    return [active_classifier.classify(marker) for marker in markers]


def _classify_hls_tag(marker: AdMarker) -> str:
    for candidate in _tag_candidates(marker):
        normalized = candidate.strip().lower()
        if normalized.startswith("#"):
            normalized = normalized[1:]
        if normalized.startswith("ext-x-cue-out"):
            return AD_START
        if normalized.startswith("ext-x-cue-in"):
            return AD_END
    return UNKNOWN


def _tag_candidates(marker: AdMarker) -> Iterable[str]:
    tag = getattr(marker, "tag", None)
    if isinstance(tag, str):
        yield tag

    tags = getattr(marker, "tags", None)
    if isinstance(tags, list):
        for value in tags:
            if isinstance(value, str):
                yield value


def _classify_scte35(fields: object) -> str:
    if not isinstance(fields, dict):
        return UNKNOWN

    command_name = fields.get("CommandName")
    if not isinstance(command_name, str):
        return UNKNOWN

    normalized_command = _normalize_phrase(command_name)
    if normalized_command == "splice insert":
        out_of_network = fields.get("OutOfNetworkIndicator")
        if isinstance(out_of_network, str) and out_of_network.strip().lower() == "true":
            return AD_START
        if out_of_network is None or isinstance(out_of_network, str):
            return AD_END
        return UNKNOWN

    if normalized_command == "time signal":
        segmentation_id = _parse_segmentation_type_id(fields.get("SegmentationTypeID"))
        if segmentation_id in _SCTE35_START_SEGMENTATION_IDS:
            return AD_START
        if segmentation_id in _SCTE35_END_SEGMENTATION_IDS:
            return AD_END

    return UNKNOWN


def _parse_segmentation_type_id(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text:
        return None

    try:
        if text.startswith("0x"):
            return int(text, 16)
        return int(text, 16)
    except ValueError:
        return None


def _classify_id3(marker: AdMarker) -> str:
    for candidate in _id3_candidates(marker):
        classification = _classify_id3_candidate(candidate)
        if classification != UNKNOWN:
            return classification
    return UNKNOWN


def _id3_candidates(marker: AdMarker) -> Iterable[str]:
    tags = getattr(marker, "tags", None)
    if isinstance(tags, list):
        for value in tags:
            if isinstance(value, str):
                yield value

    fields = getattr(marker, "fields", None)
    if not isinstance(fields, dict):
        return

    frames = fields.get("Frames")
    if not isinstance(frames, list):
        return

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        description = frame.get("Description")
        if description is not None and not isinstance(description, str):
            continue
        if isinstance(description, str):
            yield description

        text = frame.get("Text")
        if not isinstance(text, list):
            continue
        for value in text:
            if isinstance(value, str):
                yield value


def _classify_id3_candidate(candidate: str) -> str:
    normalized = _lower_text(candidate)
    if any(end_keyword in normalized for end_keyword in _ID3_END_SUBSTRINGS):
        return AD_END

    words = set(_WORD_RE.findall(normalized))
    if words & _ID3_START_WORDS:
        return AD_START

    return UNKNOWN


def _lower_text(value: object) -> str:
    return value.lower() if isinstance(value, str) else ""


def _normalize_phrase(value: str) -> str:
    return " ".join(value.replace("_", " ").split()).lower()
