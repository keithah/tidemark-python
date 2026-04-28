"""Fingerprint result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcoustIDLookupResult:
    """Normalized, redacted AcoustID match evidence."""

    acoustid_id: str | None
    recording_id: str | None
    title: str | None
    artist: str | None
    album: str | None
    score: float | None
    raw_status: str
    lookup_source: str


class AcoustIDLookupError(ValueError):
    """Redacted lookup failure with stable phase/status/sequence context."""

    def __init__(
        self,
        *,
        phase: str,
        status: str,
        sequence: int | None = None,
        detail: str = "lookup failed",
        cause: BaseException | None = None,
    ) -> None:
        self.phase = phase
        self.status = status
        self.sequence = sequence
        sequence_text = f" at sequence {sequence}" if sequence is not None else ""
        super().__init__(f"AcoustID lookup error during {phase}{sequence_text}: {status} {detail}")
        if cause is not None:
            self.__cause__ = cause


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
