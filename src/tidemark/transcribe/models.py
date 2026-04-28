"""Transcription result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WordToken:
    """One transcript word with absolute stream timestamps."""

    text: str
    start_ts: float
    end_ts: float
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptResult:
    """Transcript words plus optional engine metadata."""

    words: tuple[WordToken, ...]
    language: str | None = None
    engine: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
