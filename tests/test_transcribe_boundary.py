from __future__ import annotations

import pytest

from tidemark.audio import AudioChunk
from tidemark.transcribe import FixtureTranscriber, TranscriptResult, Transcriber, WordToken


def _chunk(*, start_ts: float = 12.5, source_url: str = "fixture://stream/private?token=secret") -> AudioChunk:
    return AudioChunk(
        pcm_bytes=b"\x00\x01",
        sample_rate=16000,
        channels=1,
        sample_format="s16le",
        segment_sequence=7,
        source_url=source_url,
        resolved_uri="file:///private/tmp/source.wav?token=secret",
        start_ts=start_ts,
        duration_seconds=1.0,
        byte_length=2,
        metadata={"private_phrase": "secret phrase"},
    )


def test_fixture_transcriber_returns_immutable_words_with_absolute_timestamps() -> None:
    transcriber: Transcriber = FixtureTranscriber(
        [
            ("hello", 0.10, 0.20, 0.91),
            ("tide", 0.25, 0.40, None),
        ],
        language="en",
        engine="fixture",
    )

    result = transcriber.transcribe(_chunk())

    assert isinstance(result, TranscriptResult)
    assert result.words == (
        WordToken(text="hello", start_ts=pytest.approx(12.60), end_ts=pytest.approx(12.70), confidence=0.91),
        WordToken(text="tide", start_ts=pytest.approx(12.75), end_ts=pytest.approx(12.90), confidence=None),
    )
    assert result.language == "en"
    assert result.engine == "fixture"
    assert result.metadata == {}
    assert result.words is not transcriber.transcribe(_chunk()).words
    with pytest.raises(AttributeError):
        result.words[0].text = "mutated"  # type: ignore[misc]


def test_fixture_transcriber_allows_empty_fixtures_and_zero_length_words() -> None:
    assert FixtureTranscriber([]).transcribe(_chunk()).words == ()

    result = FixtureTranscriber([("beep", 0.30, 0.30, 1.0)]).transcribe(_chunk(start_ts=2.0))

    assert result.words == (WordToken(text="beep", start_ts=2.30, end_ts=2.30, confidence=1.0),)


@pytest.mark.parametrize(
    ("fixture_words", "field"),
    [
        ([("", 0.10, 0.20, 0.9)], "text"),
        ([("private phrase", -0.10, 0.20, 0.9)], "start_offset"),
        ([("private phrase", 0.30, 0.20, 0.9)], "end_offset"),
        ([("private phrase", "soon", 0.20, 0.9)], "start_offset"),
        ([("private phrase", 0.10, "later", 0.9)], "end_offset"),
        ([("private phrase", 0.10, 0.20, -0.1)], "confidence"),
        ([("private phrase", 0.10, 0.20, 1.1)], "confidence"),
        ([("private phrase", 0.10, 0.20)], "fixture_words"),
    ],
)
def test_fixture_transcriber_rejects_malformed_fixtures_without_leaking_values(fixture_words: list[tuple], field: str) -> None:
    chunk = _chunk(source_url="https://example.test/private.m3u8?token=secret")

    with pytest.raises((TypeError, ValueError), match=field) as excinfo:
        FixtureTranscriber(fixture_words).transcribe(chunk)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "private phrase" not in message
    assert "example.test" not in message
    assert "token=secret" not in message
    assert "source.wav" not in message
