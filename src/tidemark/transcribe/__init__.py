"""Transcription boundary public API."""

from tidemark.transcribe.apple import AppleSpeechTranscriber, AppleSpeechTranscriptionError, AppleSpeechUnavailable
from tidemark.transcribe.fake import FixtureTranscriber
from tidemark.transcribe.models import TranscriptResult, WordToken
from tidemark.transcribe.protocol import Transcriber

DeterministicTranscriber = FixtureTranscriber

__all__ = [
    "AppleSpeechTranscriber",
    "AppleSpeechTranscriptionError",
    "AppleSpeechUnavailable",
    "DeterministicTranscriber",
    "FixtureTranscriber",
    "TranscriptResult",
    "Transcriber",
    "WordToken",
]
