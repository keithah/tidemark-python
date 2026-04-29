from __future__ import annotations

import hashlib
import importlib.util
import io
import math
import os
import struct
import sys
import wave
from collections.abc import Iterable
from pathlib import Path

import shutil
import pytest

from tidemark.audio import decode_segment_audio
from tidemark.fingerprint import (
    AcoustIDLookupError,
    AcoustIDLookupResult,
    FingerprintError,
    PyAcoustIDLookupAdapter,
    fingerprint_audio_chunk,
)
from tidemark.ingest import SegmentRecord


SECRET_VALUES = (
    "sk_live_secret_key",
    "RAW-FINGERPRINT-SECRET",
    "private backend exploded",
)


def _real_smoke_skip_reason(*, env: dict[str, str | None], acoustid_available: bool) -> str | None:
    offline = env.get("TIDEMARK_OFFLINE")
    if offline is not None and offline.strip() not in {"", "0", "false", "False", "FALSE"}:
        return "TIDEMARK_OFFLINE is set"
    api_key = env.get("ACOUSTID_API_KEY")
    if api_key is None or not api_key.strip():
        return "ACOUSTID_API_KEY is not set"
    if not acoustid_available:
        return None
    return None


def _acoustid_dependency_available() -> bool:
    return importlib.util.find_spec("acoustid") is not None


def _make_tiny_wav(path: Path) -> Path:
    sample_rate = 16000
    n = int(sample_rate * 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *[int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]))
    path.write_bytes(buf.getvalue())
    return path


def _segment_for(path: Path) -> SegmentRecord:
    data = path.read_bytes()
    return SegmentRecord(
        source_url="fixture://acoustid-real-smoke/source.m3u8",
        sequence=1,
        resolved_uri=path.as_uri(),
        local_path=str(path),
        start_ts=0.0,
        duration_seconds=1.0,
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        metadata={"source_label": "acoustid-real-smoke"},
        _loader=lambda: data,
    )


def _deterministic_backend(_sample_rate: int, _channels: int, pcmiter: Iterable[bytes]) -> tuple[float, str]:
    payload = b"".join(pcmiter)
    return (1.0, "test-fingerprint-" + hashlib.sha256(payload).hexdigest()[:16])


@pytest.mark.parametrize(
    ("env", "acoustid_available", "expected"),
    [
        ({"TIDEMARK_OFFLINE": "1", "ACOUSTID_API_KEY": "sk_live_secret_key"}, False, "TIDEMARK_OFFLINE is set"),
        ({"TIDEMARK_OFFLINE": "true", "ACOUSTID_API_KEY": "sk_live_secret_key"}, True, "TIDEMARK_OFFLINE is set"),
        ({"TIDEMARK_OFFLINE": None, "ACOUSTID_API_KEY": None}, True, "ACOUSTID_API_KEY is not set"),
        ({"TIDEMARK_OFFLINE": "0", "ACOUSTID_API_KEY": "   "}, True, "ACOUSTID_API_KEY is not set"),
        ({"TIDEMARK_OFFLINE": None, "ACOUSTID_API_KEY": "sk_live_secret_key"}, False, None),
        ({"TIDEMARK_OFFLINE": "0", "ACOUSTID_API_KEY": "sk_live_secret_key"}, True, None),
    ],
)
def test_real_smoke_skip_policy_is_key_gated_not_dependency_gated(
    env: dict[str, str | None], acoustid_available: bool, expected: str | None
) -> None:
    assert _real_smoke_skip_reason(env=env, acoustid_available=acoustid_available) == expected


def test_real_smoke_missing_dependency_with_key_is_redacted_dependency_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ACOUSTID_API_KEY", "sk_live_secret_key")
    monkeypatch.setitem(sys.modules, "acoustid", None)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(
            fingerprint_audio_chunk(
                decode_segment_audio(_segment_for(_make_tiny_wav(tmp_path / "tone.wav"))),
                backend=_deterministic_backend,
            )
        )

    assert excinfo.value.phase == "dependency"
    assert excinfo.value.status == "unavailable"
    message = str(excinfo.value)
    assert "acoustid unavailable" in message
    for secret in SECRET_VALUES:
        assert secret not in message


def test_real_acoustid_lookup_smoke_is_key_gated_and_redacted(tmp_path: Path) -> None:
    reason = _real_smoke_skip_reason(
        env={
            "TIDEMARK_OFFLINE": os.environ.get("TIDEMARK_OFFLINE"),
            "ACOUSTID_API_KEY": os.environ.get("ACOUSTID_API_KEY"),
        },
        acoustid_available=_acoustid_dependency_available(),
    )
    if reason is not None:
        pytest.skip(reason)

    media_path = _make_tiny_wav(tmp_path / "tone.wav")
    chunk = decode_segment_audio(_segment_for(media_path))
    try:
        fingerprint = fingerprint_audio_chunk(chunk)
    except FingerprintError as exc:
        pytest.fail(f"real Chromaprint backend failed with redacted error: {exc}")

    try:
        result = PyAcoustIDLookupAdapter()(fingerprint, timeout_seconds=10.0)
    except AcoustIDLookupError as exc:
        message = str(exc)
        for secret in SECRET_VALUES:
            assert secret not in message
        if exc.phase == "parse" and exc.status == "error":
            pytest.xfail("ACOUSTID_API_KEY was rejected by AcoustID; failure was redacted")
        raise

    assert isinstance(result, AcoustIDLookupResult)
    assert result.lookup_source == "acoustid"
    assert result.raw_status in {"ok", "no_match"}
    if result.raw_status == "ok":
        assert result.acoustid_id
        assert result.recording_id
        assert result.score is not None
    else:
        assert result.raw_status == "no_match"
        assert result.acoustid_id is None
        assert result.recording_id is None
