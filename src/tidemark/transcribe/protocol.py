"""Transcriber protocol boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tidemark.audio import AudioChunk
from tidemark.transcribe.models import TranscriptResult


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for audio chunk transcription adapters."""

    def transcribe(self, chunk: AudioChunk) -> TranscriptResult:
        """Transcribe decoded audio into word-level timestamps."""
