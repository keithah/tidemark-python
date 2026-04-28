"""Audio preparation models for downstream transcription boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioChunk:
    """Decoded PCM audio plus segment/source metadata for transcription handoff."""

    pcm_bytes: bytes
    sample_rate: int
    channels: int
    sample_format: str
    segment_sequence: int
    source_url: str
    resolved_uri: str
    start_ts: float
    duration_seconds: float | None
    byte_length: int
    metadata: dict[str, str] = field(default_factory=dict)
