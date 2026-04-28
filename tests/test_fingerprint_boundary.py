from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

import imageio_ffmpeg
import pytest

from tidemark.audio import AudioChunk, decode_segment_audio
from tidemark.ingest import SegmentRecord
from tidemark.fingerprint import AudioFingerprint, FingerprintError, fingerprint_audio_chunk


def _chunk(
    *,
    pcm_bytes: bytes = b"\x00\x01\x02\x03" * 8000,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_format: str = "s16le",
    duration_seconds: float | None = 2.0,
    segment_sequence: int = 11,
    source_url: str = "https://example.test/private/stream.m3u8?token=secret",
    metadata: dict[str, str] | None = None,
) -> AudioChunk:
    return AudioChunk(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        segment_sequence=segment_sequence,
        source_url=source_url,
        resolved_uri="file:///Users/alice/private/segment.ts?token=secret",
        start_ts=42.5,
        duration_seconds=duration_seconds,
        byte_length=len(pcm_bytes),
        metadata=metadata or {"source_label": "fixture-a", "private_token": "secret", "program": "news"},
    )


def _fingerprint_backend(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> str:
    payload = b"".join(pcmiter)
    return f"fp:{sample_rate}:{channels}:{hashlib.sha256(payload).hexdigest()[:16]}"


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
            "sine=frequency=660:duration=0.25:sample_rate=8000",
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


def _segment_for(data: bytes) -> SegmentRecord:
    return SegmentRecord(
        source_url="https://example.test/private/playlist.m3u8?token=secret",
        sequence=23,
        resolved_uri="file:///Users/alice/private/segment.wav?token=secret",
        local_path="/Users/alice/private/segment.wav",
        start_ts=100.25,
        duration_seconds=0.25,
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        metadata={"source_label": "fixture-decoded", "private_path": "/Users/alice/private"},
        _loader=lambda: data,
    )


def test_fingerprint_audio_chunk_returns_immutable_public_model_from_injected_backend() -> None:
    seen: dict[str, object] = {}

    def backend(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> str:
        chunks = tuple(pcmiter)
        seen["sample_rate"] = sample_rate
        seen["channels"] = channels
        seen["chunks"] = chunks
        return _fingerprint_backend(sample_rate, channels, chunks)

    result = fingerprint_audio_chunk(_chunk(), backend=backend)

    assert isinstance(result, AudioFingerprint)
    assert result.fingerprint.startswith("fp:16000:1:")
    assert result.duration_seconds == 2.0
    assert result.algorithm == "chromaprint"
    assert result.segment_sequence == 11
    assert result.source_url == "https://example.test/private/stream.m3u8?token=secret"
    assert result.start_ts == 42.5
    assert result.metadata == {"source_label": "fixture-a", "program": "news"}
    assert seen == {"sample_rate": 16000, "channels": 1, "chunks": (_chunk().pcm_bytes,)}
    with pytest.raises(AttributeError):
        result.fingerprint = "mutated"  # type: ignore[misc]


def test_fingerprint_audio_chunk_accepts_pyacoustid_style_tuple_and_derives_duration() -> None:
    pcm = b"\x00\x01" * 16000

    def backend(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> tuple[float, str]:
        assert tuple(pcmiter) == (pcm,)
        return (1.25, "tuple-fingerprint")

    result = fingerprint_audio_chunk(_chunk(pcm_bytes=pcm, duration_seconds=None), backend=backend)

    assert result.fingerprint == "tuple-fingerprint"
    assert result.duration_seconds == pytest.approx(1.0)


def test_fingerprint_audio_chunk_is_stable_for_decoded_fixture(tmp_path: Path) -> None:
    media = _make_tiny_wav(tmp_path)
    chunk = decode_segment_audio(_segment_for(media))

    first = fingerprint_audio_chunk(chunk, backend=_fingerprint_backend)
    second = fingerprint_audio_chunk(chunk, backend=_fingerprint_backend)

    assert first == second
    assert first.fingerprint
    assert first.duration_seconds == 0.25
    assert first.segment_sequence == 23
    assert first.source_url == "https://example.test/private/playlist.m3u8?token=secret"
    assert first.start_ts == 100.25
    assert first.metadata == {"source_label": "fixture-decoded"}


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"pcm_bytes": b""}, "pcm_bytes"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"channels": 0}, "channels"),
        ({"sample_format": "f32le"}, "sample_format"),
        ({"duration_seconds": -0.01}, "duration_seconds"),
    ],
)
def test_fingerprint_audio_chunk_rejects_invalid_chunks_before_backend_without_leaks(kwargs: dict[str, object], field: str) -> None:
    def backend(*args: object, **kwargs: object) -> str:  # pragma: no cover - must not be called
        raise AssertionError("backend should not run")

    with pytest.raises(FingerprintError, match=field) as excinfo:
        fingerprint_audio_chunk(_chunk(**kwargs), backend=backend)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "Fingerprint error during validation at sequence 11" in message
    assert "token=secret" not in message
    assert "Users/alice" not in message
    assert "private" not in message
    assert "\x00" not in message


def test_fingerprint_audio_chunk_accepts_pyacoustid_byte_fingerprint() -> None:
    def backend(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> bytes:
        assert tuple(pcmiter) == (_chunk().pcm_bytes,)
        return b"AQAAAA"

    result = fingerprint_audio_chunk(_chunk(), backend=backend)

    assert result.fingerprint == "AQAAAA"


@pytest.mark.parametrize("response", [b"", b"\xff", "", (1.0, ""), ("fingerprint", 1.0), (1.0,), (1.0, "fp", "extra"), object()])
def test_fingerprint_audio_chunk_rejects_malformed_backend_returns_without_values(response: object) -> None:
    with pytest.raises(FingerprintError) as excinfo:
        fingerprint_audio_chunk(_chunk(), backend=lambda *_args: response)

    message = str(excinfo.value)
    assert "Fingerprint error during backend at sequence 11" in message
    assert "fingerprint" in message
    assert "token=secret" not in message
    assert "Users/alice" not in message
    assert "extra" not in message


def test_fingerprint_audio_chunk_wraps_backend_exceptions_without_private_values() -> None:
    def backend(*_args: object) -> str:
        raise TimeoutError("/Users/alice/private/audio.wav token=secret backend exploded")

    with pytest.raises(FingerprintError) as excinfo:
        fingerprint_audio_chunk(_chunk(), backend=backend)

    message = str(excinfo.value)
    assert "Fingerprint error during backend at sequence 11" in message
    assert "backend failed" in message
    assert "backend exploded" not in message
    assert "token=secret" not in message
    assert "Users/alice" not in message


def test_fingerprint_package_imports_without_acoustid_module() -> None:
    script = """
import sys
sys.modules['acoustid'] = None
from tidemark.fingerprint import PyAcoustIDLookupAdapter, identify_fingerprint
assert PyAcoustIDLookupAdapter is not None
assert identify_fingerprint is not None
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_backend_import_is_lazy_and_missing_dependency_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "acoustid", None)

    with pytest.raises(FingerprintError) as excinfo:
        fingerprint_audio_chunk(_chunk(), backend=None)

    message = str(excinfo.value)
    assert "Fingerprint error during dependency at sequence 11" in message
    assert "acoustid unavailable" in message
    assert "token=secret" not in message
    assert "Users/alice" not in message


def test_default_backend_calls_lazy_acoustid_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, tuple[bytes, ...]]] = []
    module = ModuleType("acoustid")

    def fingerprint(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> tuple[float, str]:
        calls.append((sample_rate, channels, tuple(pcmiter)))
        return (9.9, "lazy-fingerprint")

    module.fingerprint = fingerprint  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)
    chunk = _chunk(duration_seconds=None)

    result = fingerprint_audio_chunk(chunk)

    assert result.fingerprint == "lazy-fingerprint"
    assert result.duration_seconds == pytest.approx(len(chunk.pcm_bytes) / (16000 * 1 * 2))
    assert calls == [(16000, 1, (chunk.pcm_bytes,))]
