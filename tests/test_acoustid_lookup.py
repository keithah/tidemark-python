from __future__ import annotations

import sqlite3
import sys
from types import ModuleType

import pytest

from tidemark.fingerprint import (
    AcoustIDLookupError,
    AcoustIDLookupResult,
    AudioFingerprint,
    PyAcoustIDLookupAdapter,
    identify_fingerprint,
    normalize_acoustid_lookup_response,
)
from tidemark.store import get_fingerprint_cache, get_song, insert_fingerprint_cache, insert_segment, migrate


RAW_SECRET_VALUES = (
    "sk_live_secret_key",
    "https://example.test/private/audio.wav?token=secret",
    "RAW-FINGERPRINT-SECRET",
    "private backend exploded",
    "raw-payload-secret",
)


def _fingerprint(
    *,
    value: str = "fingerprint-v1",
    sequence: int = 7,
    source_url: str = "fixture://stream",
    start_ts: float = 42.0,
    duration_seconds: float = 6.0,
) -> AudioFingerprint:
    return AudioFingerprint(
        fingerprint=value,
        duration_seconds=duration_seconds,
        algorithm="chromaprint",
        segment_sequence=sequence,
        source_url=source_url,
        start_ts=start_ts,
    )


def _conn_with_segment(*, sequence: int = 7) -> tuple[sqlite3.Connection, int]:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    segment_id = insert_segment(
        conn,
        source_url="fixture://stream",
        sequence=sequence,
        resolved_uri=f"fixture://segment-{sequence}.ts",
        local_path=None,
        start_ts=42.0,
        duration_seconds=6.0,
        byte_length=2048,
        sha256="a" * 64,
    )
    return conn, segment_id


def _adapter_result(*, lookup_source: str = "deterministic-adapter") -> AcoustIDLookupResult:
    return AcoustIDLookupResult(
        acoustid_id="acoustid-1",
        recording_id="recording-1",
        title="Matched Song",
        artist="Matched Artist",
        album="Matched Album",
        score=0.87,
        raw_status="ok",
        lookup_source=lookup_source,
    )


def _unexpected_adapter(*args: object, **kwargs: object) -> AcoustIDLookupResult:
    raise AssertionError("adapter should not be called on cache hit")


def test_identify_fingerprint_cache_hit_persists_song_without_env_import_or_adapter_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)
    conn, segment_id = _conn_with_segment()
    insert_fingerprint_cache(
        conn,
        fingerprint="fingerprint-v1",
        acoustid_id="cached-acoustid",
        recording_id="cached-recording",
        title="Cached Song",
        artist="Cached Artist",
        album="Cached Album",
        score=0.74,
        raw_status="ok",
        lookup_source="acoustid",
    )

    outcome = identify_fingerprint(
        conn,
        _fingerprint(),
        segment_id=segment_id,
        lookup_adapter=_unexpected_adapter,
        api_key="sk_live_secret_key",
    )

    assert outcome.lookup_source == "cache"
    assert outcome.cache_hit is True
    assert outcome.cache_fingerprint == "fingerprint-v1"
    assert outcome.lookup_result == AcoustIDLookupResult(
        acoustid_id="cached-acoustid",
        recording_id="cached-recording",
        title="Cached Song",
        artist="Cached Artist",
        album="Cached Album",
        score=0.74,
        raw_status="ok",
        lookup_source="cache",
    )
    stored_song = get_song(conn, outcome.song_id)
    assert stored_song is not None
    assert stored_song.segment_id == segment_id
    assert stored_song.source_url == "fixture://stream"
    assert stored_song.segment_sequence == 7
    assert stored_song.fingerprint == "fingerprint-v1"
    assert stored_song.title == "Cached Song"
    assert stored_song.lookup_source == "cache"


def test_identify_fingerprint_cache_miss_calls_adapter_once_and_persists_cache_and_song() -> None:
    conn, segment_id = _conn_with_segment()
    calls: list[tuple[AudioFingerprint, str | None, float | None]] = []

    def adapter(
        fingerprint: AudioFingerprint,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AcoustIDLookupResult:
        calls.append((fingerprint, api_key, timeout_seconds))
        return _adapter_result()

    outcome = identify_fingerprint(
        conn,
        _fingerprint(),
        segment_id=segment_id,
        lookup_adapter=adapter,
        api_key="sk_live_secret_key",
        timeout_seconds=2.5,
    )

    assert len(calls) == 1
    assert calls[0] == (_fingerprint(), "sk_live_secret_key", 2.5)
    assert outcome.lookup_source == "deterministic-adapter"
    assert outcome.cache_hit is False
    assert outcome.lookup_result == _adapter_result()
    stored_cache = get_fingerprint_cache(conn, "fingerprint-v1")
    stored_song = get_song(conn, outcome.song_id)
    assert stored_cache is not None
    assert stored_cache.acoustid_id == "acoustid-1"
    assert stored_cache.recording_id == "recording-1"
    assert stored_cache.title == "Matched Song"
    assert stored_cache.score == pytest.approx(0.87)
    assert stored_cache.raw_status == "ok"
    assert stored_cache.lookup_source == "deterministic-adapter"
    assert stored_song is not None
    assert stored_song.lookup_source == "deterministic-adapter"
    assert stored_song.title == "Matched Song"


def test_identify_fingerprint_persists_no_match_evidence_on_cache_miss() -> None:
    conn, segment_id = _conn_with_segment()

    def adapter(
        fingerprint: AudioFingerprint,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AcoustIDLookupResult:
        return AcoustIDLookupResult(
            acoustid_id=None,
            recording_id=None,
            title=None,
            artist=None,
            album=None,
            score=None,
            raw_status="no_match",
            lookup_source="fixture-no-match",
        )

    outcome = identify_fingerprint(conn, _fingerprint(), segment_id=segment_id, lookup_adapter=adapter)

    assert outcome.lookup_source == "fixture-no-match"
    assert outcome.lookup_result.raw_status == "no_match"
    stored_cache = get_fingerprint_cache(conn, "fingerprint-v1")
    stored_song = get_song(conn, outcome.song_id)
    assert stored_cache is not None
    assert stored_cache.acoustid_id is None
    assert stored_cache.raw_status == "no_match"
    assert stored_cache.lookup_source == "fixture-no-match"
    assert stored_song is not None
    assert stored_song.acoustid_id is None
    assert stored_song.lookup_source == "fixture-no-match"


def test_identify_fingerprint_repeated_cache_hits_create_song_rows_but_single_cache_row() -> None:
    conn, first_segment_id = _conn_with_segment(sequence=7)
    second_segment_id = insert_segment(
        conn,
        source_url="fixture://stream",
        sequence=8,
        resolved_uri="fixture://segment-8.ts",
        local_path=None,
        start_ts=48.0,
        duration_seconds=6.0,
        byte_length=2048,
        sha256="b" * 64,
    )
    insert_fingerprint_cache(
        conn,
        fingerprint="fingerprint-v1",
        acoustid_id="cached-acoustid",
        recording_id="cached-recording",
        title="Cached Song",
        artist="Cached Artist",
        album="Cached Album",
        score=0.74,
        raw_status="ok",
        lookup_source="acoustid",
    )

    first = identify_fingerprint(conn, _fingerprint(sequence=7, start_ts=42.0), first_segment_id, _unexpected_adapter)
    second = identify_fingerprint(conn, _fingerprint(sequence=8, start_ts=48.0), second_segment_id, _unexpected_adapter)

    assert first.song_id != second.song_id
    assert conn.execute("select count(*) from songs").fetchone()[0] == 2
    assert conn.execute("select count(*) from fingerprint_cache").fetchone()[0] == 1
    second_song = get_song(conn, second.song_id)
    assert second_song is not None
    assert second_song.segment_id == second_segment_id
    assert second_song.segment_sequence == 8
    assert second_song.start_ts == pytest.approx(48.0)


def test_identify_fingerprint_propagates_redacted_adapter_errors_without_persistence() -> None:
    conn, segment_id = _conn_with_segment()

    def adapter(
        fingerprint: AudioFingerprint,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AcoustIDLookupResult:
        raise AcoustIDLookupError(
            phase="api",
            status="timeout",
            sequence=fingerprint.segment_sequence,
            detail="adapter timeout",
            cause=TimeoutError("private backend exploded sk_live_secret_key"),
        )

    with pytest.raises(AcoustIDLookupError) as excinfo:
        identify_fingerprint(
            conn,
            _fingerprint(value="RAW-FINGERPRINT-SECRET"),
            segment_id=segment_id,
            lookup_adapter=adapter,
        )

    assert excinfo.value.phase == "api"
    assert excinfo.value.status == "timeout"
    assert excinfo.value.sequence == 7
    assert conn.execute("select count(*) from songs").fetchone()[0] == 0
    assert conn.execute("select count(*) from fingerprint_cache").fetchone()[0] == 0
    for secret in RAW_SECRET_VALUES:
        assert secret not in str(excinfo.value)


@pytest.mark.parametrize(
    ("fingerprint", "segment_id", "message"),
    [
        (_fingerprint(value=""), 1, "fingerprint"),
        (_fingerprint(), -1, "segment_id"),
        (_fingerprint(source_url=""), 1, "source_url"),
    ],
)
def test_identify_fingerprint_rejects_malformed_segment_evidence_without_adapter_call(
    fingerprint: AudioFingerprint,
    segment_id: int,
    message: str,
) -> None:
    conn, _ = _conn_with_segment()

    with pytest.raises((TypeError, ValueError), match=message):
        identify_fingerprint(conn, fingerprint, segment_id=segment_id, lookup_adapter=_unexpected_adapter)


def _successful_response() -> dict[str, object]:
    return {
        "status": "ok",
        "results": [
            {
                "id": "low-score-acoustid",
                "score": 0.42,
                "recordings": [
                    {
                        "id": "low-recording",
                        "title": "Low Result",
                        "artists": [{"name": "Ignored Artist"}],
                        "releasegroups": [{"title": "Ignored Album"}],
                    }
                ],
            },
            {
                "id": "best-acoustid",
                "score": 0.91,
                "recordings": [
                    {
                        "id": "best-recording",
                        "title": "Best Song",
                        "artists": [{"name": "Alice"}, {"name": "Bob"}],
                        "releasegroups": [{"title": "Best Album"}],
                    }
                ],
            },
        ],
    }


def test_normalize_acoustid_lookup_response_selects_highest_scoring_recording() -> None:
    result = normalize_acoustid_lookup_response(_successful_response(), lookup_source="fixture", sequence=17)

    assert result == AcoustIDLookupResult(
        acoustid_id="best-acoustid",
        recording_id="best-recording",
        title="Best Song",
        artist="Alice; Bob",
        album="Best Album",
        score=0.91,
        raw_status="ok",
        lookup_source="fixture",
    )
    with pytest.raises(AttributeError):
        result.title = "mutated"  # type: ignore[misc]


def test_normalize_acoustid_lookup_response_accepts_release_title_fallback_and_skips_blank_artists() -> None:
    response = {
        "status": "ok",
        "results": [
            {
                "id": "acoustid-1",
                "score": 0.8,
                "recordings": [
                    {
                        "id": "recording-1",
                        "title": "Song",
                        "artists": [{"name": ""}, {"name": "Solo Artist"}],
                        "releases": [{"title": "Release Album"}],
                    }
                ],
            }
        ],
    }

    result = normalize_acoustid_lookup_response(response, lookup_source="adapter", sequence=18)

    assert result.artist == "Solo Artist"
    assert result.album == "Release Album"
    assert result.raw_status == "ok"
    assert result.lookup_source == "adapter"


def test_normalize_acoustid_lookup_response_represents_empty_results_as_no_match() -> None:
    result = normalize_acoustid_lookup_response({"status": "ok", "results": []}, lookup_source="cache", sequence=19)

    assert result == AcoustIDLookupResult(
        acoustid_id=None,
        recording_id=None,
        title=None,
        artist=None,
        album=None,
        score=None,
        raw_status="no_match",
        lookup_source="cache",
    )


@pytest.mark.parametrize(
    ("response", "match"),
    [
        ("not a dict", "parse"),
        ({}, "status"),
        ({"status": "error", "error": {"message": "raw-payload-secret"}}, "status"),
        ({"status": "ok"}, "results"),
        ({"status": "ok", "results": "not-a-list"}, "results"),
        ({"status": "ok", "results": [{"id": "", "score": 0.5, "recordings": []}]}, "acoustid_id"),
        ({"status": "ok", "results": [{"id": "a", "score": "high", "recordings": []}]}, "score"),
        ({"status": "ok", "results": [{"id": "a", "score": 1.1, "recordings": []}]}, "score"),
        ({"status": "ok", "results": [{"id": "a", "score": 0.9, "recordings": "bad"}]}, "recordings"),
        ({"status": "ok", "results": [{"id": "a", "score": 0.9, "recordings": []}]}, "recordings"),
        ({"status": "ok", "results": [{"id": "a", "score": 0.9, "recordings": [{"id": "", "title": "Song"}]}]}, "recording_id"),
        ({"status": "ok", "results": [{"id": "a", "score": 0.9, "recordings": [{"id": "r", "artists": "bad"}]}]}, "artists"),
        ({"status": "ok", "results": [{"id": "a", "score": 0.9, "recordings": [{"id": "r", "releases": "bad"}]}]}, "releases"),
    ],
)
def test_normalize_acoustid_lookup_response_rejects_malformed_inputs_with_redacted_errors(
    response: object, match: str
) -> None:
    with pytest.raises(AcoustIDLookupError, match=match) as excinfo:
        normalize_acoustid_lookup_response(response, lookup_source="acoustid", sequence=20)

    message = str(excinfo.value)
    assert "AcoustID lookup error during parse at sequence 20" in message
    for secret in RAW_SECRET_VALUES:
        assert secret not in message


def test_acoustid_lookup_error_is_redacted_and_exposes_stable_context() -> None:
    err = AcoustIDLookupError(
        phase="backend",
        status="timeout",
        sequence=21,
        detail="backend failed",
        cause=TimeoutError("private backend exploded sk_live_secret_key"),
    )

    assert err.phase == "backend"
    assert err.status == "timeout"
    assert err.sequence == 21
    assert str(err) == "AcoustID lookup error during backend at sequence 21: timeout backend failed"
    for secret in RAW_SECRET_VALUES:
        assert secret not in str(err)


def test_pyacoustid_lookup_adapter_imports_dependency_lazily_and_normalizes_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, float, list[str], float | None]] = []
    module = ModuleType("acoustid")

    class WebServiceError(Exception):
        pass

    def lookup(
        api_key: str,
        fingerprint: str,
        duration: float,
        *,
        meta: list[str],
        timeout: float | None,
    ) -> dict[str, object]:
        calls.append((api_key, fingerprint, duration, meta, timeout))
        return _successful_response()

    module.WebServiceError = WebServiceError  # type: ignore[attr-defined]
    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)

    result = PyAcoustIDLookupAdapter()(
        _fingerprint(value="RAW-FINGERPRINT-SECRET", duration_seconds=8.5),
        api_key="sk_live_secret_key",
        timeout_seconds=3.25,
    )

    assert result.lookup_source == "acoustid"
    assert result.acoustid_id == "best-acoustid"
    assert result.recording_id == "best-recording"
    assert result.score == pytest.approx(0.91)
    assert calls == [
        (
            "sk_live_secret_key",
            "RAW-FINGERPRINT-SECRET",
            8.5,
            ["recordings", "releases"],
            3.25,
        )
    ]


def test_pyacoustid_lookup_adapter_reads_env_key_only_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    module = ModuleType("acoustid")

    def lookup(api_key: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append(api_key)
        return {"status": "ok", "results": []}

    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)
    monkeypatch.setenv("ACOUSTID_API_KEY", "sk_live_secret_key")

    result = PyAcoustIDLookupAdapter()(_fingerprint(), timeout_seconds=None)

    assert calls == ["sk_live_secret_key"]
    assert result.raw_status == "no_match"
    assert result.lookup_source == "acoustid"


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_pyacoustid_lookup_adapter_rejects_missing_or_blank_key_before_import(
    api_key: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "acoustid", None)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(_fingerprint(value="RAW-FINGERPRINT-SECRET"), api_key=api_key)

    assert excinfo.value.phase == "auth"
    assert excinfo.value.status == "missing_key"
    assert excinfo.value.sequence == 7
    message = str(excinfo.value)
    assert "API key unavailable" in message
    for secret in RAW_SECRET_VALUES:
        assert secret not in message


def test_pyacoustid_lookup_adapter_rejects_invalid_duration_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("acoustid")

    def lookup(*_args: object, **_kwargs: object) -> dict[str, object]:  # pragma: no cover - must not run
        raise AssertionError("lookup should not be called")

    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(_fingerprint(duration_seconds=0), api_key="sk_live_secret_key")

    assert excinfo.value.phase == "validation"
    assert excinfo.value.status == "invalid_duration"
    assert "duration" in str(excinfo.value)
    assert "sk_live_secret_key" not in str(excinfo.value)


def test_pyacoustid_lookup_adapter_reports_missing_dependency_without_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACOUSTID_API_KEY", "sk_live_secret_key")
    monkeypatch.setitem(sys.modules, "acoustid", None)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(_fingerprint(value="RAW-FINGERPRINT-SECRET"))

    assert excinfo.value.phase == "dependency"
    assert excinfo.value.status == "unavailable"
    assert excinfo.value.sequence == 7
    message = str(excinfo.value)
    assert "acoustid unavailable" in message
    for secret in RAW_SECRET_VALUES:
        assert secret not in message


def test_pyacoustid_lookup_adapter_wraps_web_service_errors_without_backend_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("acoustid")

    class WebServiceError(Exception):
        pass

    def lookup(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise WebServiceError("private backend exploded sk_live_secret_key")

    module.WebServiceError = WebServiceError  # type: ignore[attr-defined]
    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(_fingerprint(value="RAW-FINGERPRINT-SECRET"), api_key="sk_live_secret_key")

    assert excinfo.value.phase == "service"
    assert excinfo.value.status == "web_service_error"
    message = str(excinfo.value)
    assert "service lookup failed" in message
    for secret in RAW_SECRET_VALUES:
        assert secret not in message


@pytest.mark.parametrize(
    ("backend_error", "expected_phase", "expected_status"),
    [
        (TimeoutError("private backend exploded sk_live_secret_key"), "timeout", "timeout"),
        (OSError("private backend exploded sk_live_secret_key"), "service", "network_error"),
    ],
)
def test_pyacoustid_lookup_adapter_wraps_timeout_and_network_errors_without_backend_text(
    backend_error: BaseException,
    expected_phase: str,
    expected_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("acoustid")

    def lookup(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise backend_error

    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(_fingerprint(value="RAW-FINGERPRINT-SECRET"), api_key="sk_live_secret_key")

    assert excinfo.value.phase == expected_phase
    assert excinfo.value.status == expected_status
    message = str(excinfo.value)
    for secret in RAW_SECRET_VALUES:
        assert secret not in message


def test_pyacoustid_lookup_adapter_routes_malformed_response_through_redacted_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("acoustid")

    def lookup(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "ok", "results": "raw-payload-secret"}

    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)

    with pytest.raises(AcoustIDLookupError) as excinfo:
        PyAcoustIDLookupAdapter()(_fingerprint(), api_key="sk_live_secret_key")

    assert excinfo.value.phase == "parse"
    assert excinfo.value.status == "malformed"
    assert "raw-payload-secret" not in str(excinfo.value)


def test_pyacoustid_lookup_adapter_returns_no_match_for_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("acoustid")

    def lookup(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "ok", "results": []}

    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)

    result = PyAcoustIDLookupAdapter()(_fingerprint(), api_key="sk_live_secret_key")

    assert result == AcoustIDLookupResult(
        acoustid_id=None,
        recording_id=None,
        title=None,
        artist=None,
        album=None,
        score=None,
        raw_status="no_match",
        lookup_source="acoustid",
    )


def test_pyacoustid_lookup_adapter_preserves_low_confidence_match_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("acoustid")

    def lookup(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "results": [
                {
                    "id": "low-acoustid",
                    "score": 0.03,
                    "recordings": [{"id": "low-recording", "title": "Low Confidence"}],
                }
            ],
        }

    module.lookup = lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acoustid", module)

    result = PyAcoustIDLookupAdapter()(_fingerprint(), api_key="sk_live_secret_key")

    assert result.acoustid_id == "low-acoustid"
    assert result.recording_id == "low-recording"
    assert result.title == "Low Confidence"
    assert result.score == pytest.approx(0.03)
    assert result.raw_status == "ok"
