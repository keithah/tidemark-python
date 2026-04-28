"""Fingerprint boundary public API."""

from tidemark.fingerprint.generator import FingerprintError, fingerprint_audio_chunk
from tidemark.fingerprint.models import AudioFingerprint
from tidemark.fingerprint.retention import RetainedAudioFile, RetentionError, write_retained_audio

__all__ = [
    "AudioFingerprint",
    "FingerprintError",
    "RetainedAudioFile",
    "RetentionError",
    "fingerprint_audio_chunk",
    "write_retained_audio",
]
