"""Pure AcoustID lookup response normalization."""

from __future__ import annotations

from numbers import Real
from typing import Any

from tidemark.fingerprint.models import AcoustIDLookupError, AcoustIDLookupResult


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
