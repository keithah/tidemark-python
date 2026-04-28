"""Fingerprint boundary public API."""

from tidemark.fingerprint.generator import FingerprintError, fingerprint_audio_chunk
from tidemark.fingerprint.models import AudioFingerprint

__all__ = ["AudioFingerprint", "FingerprintError", "fingerprint_audio_chunk"]
