"""Marker data models.

Exact Go timestamp string parity is deferred to the later output-parity slices; this
foundation keeps timestamps deterministic as floats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdMarker:
    """Go-compatible ad marker serialization contract."""

    type: str
    classification: str
    source: str
    tag: str | None = None
    pts: float | None = None
    segment: int | None = None
    break_duration: float | None = None
    raw_base64: str | None = None
    command: dict[str, Any] | None = None
    descriptors: list[dict[str, Any]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return only the stable capitalized keys consumed by tidemark-go users."""
        return {
            "Type": self.type,
            "Classification": self.classification,
            "Source": self.source,
            "Tag": self.tag,
            "PTS": self.pts,
            "Segment": self.segment,
            "RawBase64": self.raw_base64,
            "Command": self.command,
            "Descriptors": self.descriptors,
            "Tags": self.tags,
            "Fields": self.fields,
            "Timestamp": float(self.timestamp),
        }

    def to_json(self) -> str:
        """Serialize using the explicit Go-compatible key contract."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)
