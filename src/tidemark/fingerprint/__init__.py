"""Fingerprint boundary public API."""

from tidemark.fingerprint.generator import FingerprintError, fingerprint_audio_chunk
from tidemark.fingerprint.lookup import (
    AcoustIDLookupAdapter,
    PyAcoustIDLookupAdapter,
    identify_fingerprint,
    normalize_acoustid_lookup_response,
)
from tidemark.fingerprint.models import (
    AcoustIDLookupError,
    AcoustIDLookupResult,
    AudioFingerprint,
    FingerprintIdentificationResult,
)
from tidemark.fingerprint.retention import RetainedAudioFile, RetentionError, write_retained_audio

__all__ = [
    "AcoustIDLookupAdapter",
    "AcoustIDLookupError",
    "AcoustIDLookupResult",
    "AudioFingerprint",
    "FingerprintError",
    "FingerprintIdentificationResult",
    "PyAcoustIDLookupAdapter",
    "RetainedAudioFile",
    "RetentionError",
    "fingerprint_audio_chunk",
    "identify_fingerprint",
    "normalize_acoustid_lookup_response",
    "write_retained_audio",
]
