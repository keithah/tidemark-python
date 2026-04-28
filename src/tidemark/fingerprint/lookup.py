"""Pure AcoustID lookup response normalization and cache-first orchestration."""

from __future__ import annotations

import sqlite3
from numbers import Real
from typing import Any, Protocol

from tidemark.fingerprint.models import (
    AcoustIDLookupError,
    AcoustIDLookupResult,
    AudioFingerprint,
    FingerprintIdentificationResult,
)
from tidemark.store import (
    FingerprintCacheRecord,
    get_fingerprint_cache,
    insert_fingerprint_cache,
    insert_song,
)


class AcoustIDLookupAdapter(Protocol):
    """Callable boundary for lazy AcoustID-compatible lookup adapters."""

    def __call__(
        self,
        fingerprint: AudioFingerprint,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AcoustIDLookupResult:
        """Return normalized lookup evidence for one fingerprint."""


def identify_fingerprint(
    conn: sqlite3.Connection,
    fingerprint: AudioFingerprint,
    segment_id: int,
    lookup_adapter: AcoustIDLookupAdapter,
    *,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
) -> FingerprintIdentificationResult:
    """Identify and persist one fingerprint using the cache before adapter paths.

    Cache hits never consult API keys, construct adapters, import optional
    dependencies, or perform network work. Cache misses call the injected
    adapter once, persist normalized lookup evidence, and then persist a song
    evidence row for the current segment context.
    """
    if not isinstance(fingerprint, AudioFingerprint):
        raise TypeError("identify_fingerprint() fingerprint must be an AudioFingerprint")
    _validate_segment_context(fingerprint=fingerprint, segment_id=segment_id)

    cached = get_fingerprint_cache(conn, fingerprint.fingerprint)
    if cached is not None:
        result = _result_from_cache(cached)
        song_id = _insert_song_evidence(
            conn,
            fingerprint=fingerprint,
            segment_id=segment_id,
            result=result,
            lookup_source="cache",
        )
        return FingerprintIdentificationResult(
            lookup_source="cache",
            cache_hit=True,
            cache_fingerprint=cached.fingerprint,
            song_id=song_id,
            lookup_result=result,
        )

    try:
        result = lookup_adapter(fingerprint, api_key=api_key, timeout_seconds=timeout_seconds)
    except AcoustIDLookupError:
        raise
    except TimeoutError as exc:
        raise AcoustIDLookupError(
            phase="timeout",
            status="timeout",
            sequence=fingerprint.segment_sequence,
            detail="adapter timed out",
            cause=exc,
        ) from exc
    except Exception as exc:
        raise AcoustIDLookupError(
            phase="adapter",
            status="error",
            sequence=fingerprint.segment_sequence,
            detail="adapter failed",
            cause=exc,
        ) from exc

    if not isinstance(result, AcoustIDLookupResult):
        raise AcoustIDLookupError(
            phase="adapter",
            status="malformed",
            sequence=fingerprint.segment_sequence,
            detail="adapter returned invalid result",
        )

    insert_fingerprint_cache(
        conn,
        fingerprint=fingerprint.fingerprint,
        acoustid_id=result.acoustid_id,
        recording_id=result.recording_id,
        title=result.title,
        artist=result.artist,
        album=result.album,
        score=result.score,
        raw_status=result.raw_status,
        lookup_source=result.lookup_source,
    )
    song_id = _insert_song_evidence(
        conn,
        fingerprint=fingerprint,
        segment_id=segment_id,
        result=result,
        lookup_source=result.lookup_source,
    )
    return FingerprintIdentificationResult(
        lookup_source=result.lookup_source,
        cache_hit=False,
        cache_fingerprint=fingerprint.fingerprint,
        song_id=song_id,
        lookup_result=result,
    )


def _validate_segment_context(*, fingerprint: AudioFingerprint, segment_id: int) -> None:
    if not isinstance(segment_id, int) or isinstance(segment_id, bool):
        raise TypeError("identify_fingerprint() segment_id must be an integer")
    if segment_id < 0:
        raise ValueError("identify_fingerprint() segment_id must be >= 0")
    if not isinstance(fingerprint.source_url, str) or not fingerprint.source_url.strip():
        raise ValueError("identify_fingerprint() source_url must be a non-empty string")
    if not isinstance(fingerprint.segment_sequence, int) or isinstance(fingerprint.segment_sequence, bool):
        raise TypeError("identify_fingerprint() segment_sequence must be an integer")
    if fingerprint.segment_sequence < 0:
        raise ValueError("identify_fingerprint() segment_sequence must be >= 0")
    if not isinstance(fingerprint.start_ts, Real) or isinstance(fingerprint.start_ts, bool):
        raise TypeError("identify_fingerprint() start_ts must be a number")
    if float(fingerprint.start_ts) < 0:
        raise ValueError("identify_fingerprint() start_ts must be >= 0")
    if not isinstance(fingerprint.duration_seconds, Real) or isinstance(fingerprint.duration_seconds, bool):
        raise TypeError("identify_fingerprint() duration_seconds must be a number")
    if float(fingerprint.duration_seconds) < 0:
        raise ValueError("identify_fingerprint() duration_seconds must be >= 0")


def _result_from_cache(cached: FingerprintCacheRecord) -> AcoustIDLookupResult:
    return AcoustIDLookupResult(
        acoustid_id=cached.acoustid_id,
        recording_id=cached.recording_id,
        title=cached.title,
        artist=cached.artist,
        album=cached.album,
        score=cached.score,
        raw_status=cached.raw_status or "unknown",
        lookup_source="cache",
    )


def _insert_song_evidence(
    conn: sqlite3.Connection,
    *,
    fingerprint: AudioFingerprint,
    segment_id: int,
    result: AcoustIDLookupResult,
    lookup_source: str,
) -> int:
    return insert_song(
        conn,
        segment_id=segment_id,
        source_url=fingerprint.source_url,
        segment_sequence=fingerprint.segment_sequence,
        start_ts=fingerprint.start_ts,
        duration_seconds=fingerprint.duration_seconds,
        fingerprint=fingerprint.fingerprint,
        acoustid_id=result.acoustid_id,
        recording_id=result.recording_id,
        title=result.title,
        artist=result.artist,
        album=result.album,
        score=result.score,
        lookup_source=lookup_source,
    )


def normalize_acoustid_lookup_response(
    response: object,
    *,
    lookup_source: str,
    sequence: int | None = None,
) -> AcoustIDLookupResult:
    """Normalize a raw AcoustID-like lookup response without retaining raw payloads.

    The parser is intentionally strict: malformed fields raise a redacted
    ``AcoustIDLookupError`` instead of guessing, clamping scores, or leaking raw
    response content into diagnostics. Empty result sets are represented as a
    typed no-match result so downstream cache/orchestration code can persist a
    deterministic miss without treating it as parser failure.
    """
    if not isinstance(response, dict):
        raise _parse_error(sequence=sequence, status="malformed", detail="response must be object")

    raw_status = response.get("status")
    if not isinstance(raw_status, str) or raw_status.strip() == "":
        raise _parse_error(sequence=sequence, status="malformed", detail="missing status")
    status = raw_status.strip()
    if status != "ok":
        raise _parse_error(sequence=sequence, status=status, detail="service status not ok")

    if "results" not in response:
        raise _parse_error(sequence=sequence, status="malformed", detail="missing results")
    results = response["results"]
    if not isinstance(results, list):
        raise _parse_error(sequence=sequence, status="malformed", detail="results must be list")
    if not results:
        return AcoustIDLookupResult(
            acoustid_id=None,
            recording_id=None,
            title=None,
            artist=None,
            album=None,
            score=None,
            raw_status="no_match",
            lookup_source=lookup_source,
        )

    candidates = [_result_candidate(item, index=index, sequence=sequence) for index, item in enumerate(results)]
    best = max(candidates, key=lambda item: item["sort_score"])
    return AcoustIDLookupResult(
        acoustid_id=best["acoustid_id"],
        recording_id=best["recording_id"],
        title=best["title"],
        artist=best["artist"],
        album=best["album"],
        score=best["score"],
        raw_status=status,
        lookup_source=lookup_source,
    )


def _result_candidate(result: object, *, index: int, sequence: int | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise _parse_error(sequence=sequence, status="malformed", detail=f"results[{index}] must be object")

    acoustid_id = _required_text(result.get("id"), field="acoustid_id", sequence=sequence)
    score = _score(result.get("score"), sequence=sequence)
    recordings = result.get("recordings")
    if not isinstance(recordings, list):
        raise _parse_error(sequence=sequence, status="malformed", detail="recordings must be list")
    if not recordings:
        raise _parse_error(sequence=sequence, status="no_match", detail="no recordings")

    normalized_recordings = [
        _recording_candidate(recording, result_score=score, sequence=sequence) for recording in recordings
    ]
    best_recording = max(normalized_recordings, key=lambda item: item["sort_score"])
    return {
        "acoustid_id": acoustid_id,
        "recording_id": best_recording["recording_id"],
        "title": best_recording["title"],
        "artist": best_recording["artist"],
        "album": best_recording["album"],
        "score": score,
        "sort_score": best_recording["sort_score"],
    }


def _recording_candidate(recording: object, *, result_score: float, sequence: int | None) -> dict[str, Any]:
    if not isinstance(recording, dict):
        raise _parse_error(sequence=sequence, status="malformed", detail="recording must be object")

    recording_id = _required_text(recording.get("id"), field="recording_id", sequence=sequence)
    recording_score = recording.get("score", result_score)
    sort_score = _score(recording_score, sequence=sequence) if "score" in recording else result_score

    return {
        "recording_id": recording_id,
        "title": _optional_text(recording.get("title"), field="title", sequence=sequence),
        "artist": _artist(recording.get("artists"), sequence=sequence),
        "album": _album(recording, sequence=sequence),
        "sort_score": sort_score,
    }


def _score(value: object, *, sequence: int | None) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise _parse_error(sequence=sequence, status="malformed", detail="invalid score")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise _parse_error(sequence=sequence, status="malformed", detail="score out of range")
    return score


def _required_text(value: object, *, field: str, sequence: int | None) -> str:
    text = _optional_text(value, field=field, sequence=sequence)
    if text is None:
        raise _parse_error(sequence=sequence, status="malformed", detail=f"missing {field}")
    return text


def _optional_text(value: object, *, field: str, sequence: int | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _parse_error(sequence=sequence, status="malformed", detail=f"invalid {field}")
    text = value.strip()
    return text or None


def _artist(value: object, *, sequence: int | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise _parse_error(sequence=sequence, status="malformed", detail="artists must be list")

    names: list[str] = []
    for artist in value:
        if not isinstance(artist, dict):
            raise _parse_error(sequence=sequence, status="malformed", detail="artist must be object")
        name = _optional_text(artist.get("name"), field="artist", sequence=sequence)
        if name is not None:
            names.append(name)
    return "; ".join(names) if names else None


def _album(recording: dict[str, object], *, sequence: int | None) -> str | None:
    for field in ("releasegroups", "releases"):
        value = recording.get(field)
        if value is None:
            continue
        if not isinstance(value, list):
            raise _parse_error(sequence=sequence, status="malformed", detail=f"{field} must be list")
        for release in value:
            if not isinstance(release, dict):
                raise _parse_error(sequence=sequence, status="malformed", detail=f"{field} item must be object")
            title = _optional_text(release.get("title"), field="album", sequence=sequence)
            if title is not None:
                return title
    return None


def _parse_error(*, sequence: int | None, status: str, detail: str) -> AcoustIDLookupError:
    return AcoustIDLookupError(phase="parse", status=status, sequence=sequence, detail=detail)
