"""Fingerprint result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioFingerprint:
    """Public fingerprint plus segment context for storage and lookup."""

    fingerprint: str
    duration_seconds: float
    algorithm: str
    segment_sequence: int
    source_url: str
    start_ts: float
    metadata: dict[str, str] = field(default_factory=dict)
