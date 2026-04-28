"""Fingerprint boundary public API."""

from tidemark.fingerprint.generator import FingerprintError, fingerprint_audio_chunk
from tidemark.fingerprint.lookup import normalize_acoustid_lookup_response
from tidemark.fingerprint.models import AcoustIDLookupError, AcoustIDLookupResult, AudioFingerprint
from tidemark.fingerprint.retention import RetainedAudioFile, RetentionError, write_retained_audio

__all__ = [
    "AcoustIDLookupError",
    "AcoustIDLookupResult",
    "AudioFingerprint",
    "FingerprintError",
    "RetainedAudioFile",
    "RetentionError",
    "fingerprint_audio_chunk",
    "normalize_acoustid_lookup_response",
    "write_retained_audio",
]
