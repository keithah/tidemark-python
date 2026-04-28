"""Deterministic fixture-backed transcriber."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable

from tidemark.audio import AudioChunk
from tidemark.transcribe.models import TranscriptResult, WordToken

FixtureWord = tuple[str, float, float, float | None]


@dataclass(frozen=True)
class FixtureTranscriber:
    """Transcriber that maps fixture-relative offsets to absolute stream times."""

    fixture_words: Iterable[FixtureWord]
    language: str | None = None
    engine: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_words", tuple(_validate_fixture_words(self.fixture_words)))

    def transcribe(self, chunk: AudioChunk) -> TranscriptResult:
        words = tuple(
            WordToken(
                text=text,
                start_ts=chunk.start_ts + start_offset,
                end_ts=chunk.start_ts + end_offset,
                confidence=confidence,
            )
            for text, start_offset, end_offset, confidence in self.fixture_words
        )
        return TranscriptResult(words=words, language=self.language, engine=self.engine)


def _validate_fixture_words(fixture_words: Iterable[FixtureWord]) -> tuple[FixtureWord, ...]:
    validated: list[FixtureWord] = []
    for item in fixture_words:
        if not isinstance(item, tuple) or len(item) != 4:
            raise TypeError("fixture_words item must contain text, start_offset, end_offset, confidence")
        text, start_offset, end_offset, confidence = item
        if not isinstance(text, str) or text == "":
            raise ValueError("text must be a non-empty string")
        if not _is_number(start_offset):
            raise TypeError("start_offset must be numeric")
        if not _is_number(end_offset):
            raise TypeError("end_offset must be numeric")
        if start_offset < 0:
            raise ValueError("start_offset must be >= 0")
        if end_offset < start_offset:
            raise ValueError("end_offset must be >= start_offset")
        if confidence is not None:
            if not _is_number(confidence):
                raise TypeError("confidence must be numeric or None")
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError("confidence must be between 0 and 1")
        validated.append((text, float(start_offset), float(end_offset), None if confidence is None else float(confidence)))
    return tuple(validated)


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
