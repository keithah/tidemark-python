from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.audio import AudioChunk, AudioDecodeError, decode_segment_audio
from tidemark.ingest.segments import SegmentRecord


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


def _make_tiny_wav(tmp_path: Path) -> bytes:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output = tmp_path / "tiny.wav"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25:sample_rate=8000",
            "-ac",
            "1",
            "-y",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return output.read_bytes()


def test_decode_segment_audio_returns_mono_16khz_pcm_chunk(tmp_path: Path) -> None:
    media = _make_tiny_wav(tmp_path)
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


def test_decode_segment_audio_preserves_zero_start_timestamp_for_very_short_fixture(tmp_path: Path) -> None:
    media = _make_tiny_wav(tmp_path)
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


def test_decode_segment_audio_wraps_ffmpeg_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    segment = _segment_for(b"abc", source_url="https://cdn.example.test/live.ts?token=secret", sequence=9)

    def fail_lookup() -> str:
        raise RuntimeError("/Users/alice/private/ffmpeg missing")

    monkeypatch.setattr("tidemark.audio.decoder.imageio_ffmpeg.get_ffmpeg_exe", fail_lookup)

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment)

    message = str(excinfo.value)
    assert "Audio decode error during ffmpeg at sequence 9" in message
    assert "unavailable" in message
    assert "Users/alice" not in message
    assert "ffmpeg missing" not in message


def test_decode_segment_audio_wraps_subprocess_failure_without_stderr_or_command(monkeypatch: pytest.MonkeyPatch) -> None:
    media = b"abc"
    segment = _segment_for(media, source_url="https://cdn.example.test/live.ts?token=secret", sequence=4)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=["/Users/alice/private/ffmpeg", "-i", "https://cdn.example.test/live.ts?token=secret"],
            returncode=1,
            stdout=b"",
            stderr=b"ffmpeg stderr /Users/alice/private/file.ts?token=secret Invalid data found",
        )

    monkeypatch.setattr("tidemark.audio.decoder.subprocess.run", fake_run)

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment)

    message = str(excinfo.value)
    assert "Audio decode error during decode at sequence 4" in message
    assert "ffmpeg returned non-zero status" in message
    assert "Users/alice" not in message
    assert "token=secret" not in message
    assert "Invalid data" not in message
    assert "-i" not in message


def test_decode_segment_audio_wraps_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    segment = _segment_for(b"abc", source_url="https://cdn.example.test/live.ts?token=secret", sequence=5)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(
            cmd=["/Users/alice/private/ffmpeg", "https://cdn.example.test/live.ts?token=secret"],
            timeout=1.0,
            stderr=b"/Users/alice/private/file.ts?token=secret",
        )

    monkeypatch.setattr("tidemark.audio.decoder.subprocess.run", fake_run)

    with pytest.raises(AudioDecodeError) as excinfo:
        decode_segment_audio(segment, timeout_seconds=1.0)

    message = str(excinfo.value)
    assert "Audio decode error during decode at sequence 5" in message
    assert "ffmpeg timed out" in message
    assert "Users/alice" not in message
    assert "token=secret" not in message


def test_decode_segment_audio_command_helper_is_inspectable() -> None:
    command = decode_segment_audio.build_ffmpeg_command("/usr/bin/ffmpeg")

    assert command[:3] == ["/usr/bin/ffmpeg", "-hide_banner", "-loglevel"]
    assert "pipe:0" in command
    assert command[-2:] == ["s16le", "pipe:1"]
    assert "16000" in command
