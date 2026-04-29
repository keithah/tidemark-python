from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from pathlib import Path

import av
import pytest

from tidemark.audio import AudioChunk, AudioDecodeError, decode_segment_audio
from tidemark.ingest import SegmentRecord


def _segment_for(data: bytes, *, source_url: str = "https://example.test/live/playlist.m3u8?token=secret", sequence: int = 7) -> SegmentRecord:
    return SegmentRecord(
        source_url=source_url,
        sequence=sequence,
        resolved_uri="file:///private/tmp/tidemark/private-segment.wav?signature=secret",
        local_path="/private/tmp/tidemark/private-segment.wav",
        start_ts=0.0,
        duration_seconds=0.25,
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        metadata={"source_label": "fixture-a", "private_path": "/Users/alice/secret/input.wav"},
        _loader=lambda: data,
    )


def _make_tiny_wav() -> bytes:
    """Generate a tiny sine-wave WAV using stdlib (no external tools required)."""
    sample_rate = 8000
    duration = 0.25
    frequency = 440.0
    n_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = struct.pack(
            f"<{n_samples}h",
            *[int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(n_samples)],
        )
        wf.writeframes(frames)
    return buf.getvalue()


def test_decode_segment_audio_returns_mono_16khz_pcm_chunk() -> None:
    media = _make_tiny_wav()
    segment = _segment_for(media)

    chunk = decode_segment_audio(segment)

    assert isinstance(chunk, AudioChunk)
    assert chunk.pcm_bytes
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.sample_format == "s16le"
    assert chunk.segment_sequence == 7
    assert chunk.source_url == segment.source_url
    assert chunk.resolved_uri == segment.resolved_uri
    assert chunk.start_ts == 0.0
    assert chunk.duration_seconds == 0.25
    assert chunk.byte_length == len(chunk.pcm_bytes)
    assert chunk.metadata == {"source_label": "fixture-a"}
    assert not hasattr(chunk, "transcript")
    assert not hasattr(chunk, "search_text")


def test_decode_segment_audio_preserves_zero_start_timestamp_for_very_short_fixture() -> None:
    media = _make_tiny_wav()
    segment = _segment_for(media, sequence=0)

    chunk = decode_segment_audio(segment)

    assert chunk.segment_sequence == 0
    assert chunk.start_ts == 0.0
    assert chunk.duration_seconds == segment.duration_seconds
    assert chunk.pcm_bytes != media


@pytest.mark.parametrize("payload", [b"", b"not-media-bytes?token=secret"])
def test_decode_segment_audio_rejects_malformed_bytes_without_leaking_details(payload: bytes) -> None:
    segment = _segment_for(payload, source_url="https://cdn.example.test/live.ts?token=super-secret", sequence=3)

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment)

    message = str(excinfo.value)
    assert "Audio decode error during decode at sequence 3" in message
    assert "token=super-secret" not in message
    assert "not-media-bytes" not in message
    assert "private-segment" not in message
    assert "Invalid data" not in message


def test_decode_segment_audio_reports_missing_metadata_without_private_values() -> None:
    data = b"abc"
    segment = SegmentRecord(
        source_url="https://cdn.example.test/live.ts?token=secret",
        sequence=1,
        resolved_uri="file:///Users/alice/private/live.ts?token=secret",
        local_path="/Users/alice/private/live.ts",
        start_ts=0.0,
        duration_seconds=1.0,
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        metadata=None,
        _loader=None,
    )

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment)

    message = str(excinfo.value)
    assert "Audio decode error during load at sequence 1" in message
    assert "no segment bytes available" in message
    assert "Users/alice" not in message
    assert "token=secret" not in message


def test_decode_segment_audio_wraps_pyav_error_without_leaking_details(monkeypatch: pytest.MonkeyPatch) -> None:
    segment = _segment_for(b"abc", source_url="https://cdn.example.test/live.ts?token=secret", sequence=9)

    def fail_open(*args: object, **kwargs: object) -> None:
        raise av.AVError(-1, "Invalid data found when processing input /Users/alice/private/file.ts?token=secret")

    monkeypatch.setattr("tidemark.audio.decoder.av.open", fail_open)

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment)

    message = str(excinfo.value)
    assert "Audio decode error during decode at sequence 9" in message
    assert "decode failed" in message
    assert "Users/alice" not in message
    assert "token=secret" not in message
    assert "Invalid data" not in message


def test_decode_segment_audio_wraps_unexpected_exception_without_leaking_details(monkeypatch: pytest.MonkeyPatch) -> None:
    segment = _segment_for(b"abc", source_url="https://cdn.example.test/live.ts?token=secret", sequence=4)

    def fail_open(*args: object, **kwargs: object) -> None:
        raise RuntimeError("internal error at /Users/alice/private/file.ts?token=secret")

    monkeypatch.setattr("tidemark.audio.decoder.av.open", fail_open)

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment)

    message = str(excinfo.value)
    assert "Audio decode error during decode at sequence 4" in message
    assert "Users/alice" not in message
    assert "token=secret" not in message
    assert "internal error" not in message
