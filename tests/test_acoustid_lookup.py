from __future__ import annotations

import pytest

from tidemark.fingerprint import AcoustIDLookupError, AcoustIDLookupResult, normalize_acoustid_lookup_response


RAW_SECRET_VALUES = (
    "sk_live_secret_key",
    "https://example.test/private/audio.wav?token=secret",
    "RAW-FINGERPRINT-SECRET",
    "private backend exploded",
    "raw-payload-secret",
)


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
