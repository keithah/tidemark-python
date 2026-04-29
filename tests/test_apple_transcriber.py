from __future__ import annotations

import pytest

from tidemark.audio import AudioChunk
from tidemark.transcribe.apple import AppleSpeechTranscriber, AppleSpeechUnavailable, _words_from_result


def _chunk() -> AudioChunk:
    return AudioChunk(
        pcm_bytes=b"\x00\x00" * 160,
        sample_rate=16000,
        channels=1,
        sample_format="s16le",
        segment_sequence=7,
        source_url="https://example.test/live.m3u8?token=secret",
        resolved_uri="https://cdn.example.test/seg7.ts",
        start_ts=12.0,
        duration_seconds=0.01,
        byte_length=320,
        metadata={},
    )


def test_apple_speech_transcriber_is_unavailable_off_macos(monkeypatch) -> None:
    monkeypatch.setattr("tidemark.transcribe.apple.platform.system", lambda: "Linux")

    with pytest.raises(AppleSpeechUnavailable) as excinfo:
        AppleSpeechTranscriber()

    assert str(excinfo.value) == "apple speech is only available on macOS"


class FakeSegment:
    def __init__(self, text: str, timestamp: float, duration: float, confidence: float) -> None:
        self._text = text
        self._timestamp = timestamp
        self._duration = duration
        self._confidence = confidence

    def substring(self) -> str:
        return self._text

    def timestamp(self) -> float:
        return self._timestamp

    def duration(self) -> float:
        return self._duration

    def confidence(self) -> float:
        return self._confidence


class FakeTranscription:
    def segments(self):
        return [FakeSegment(" hello ", 0.25, 0.5, 0.9), FakeSegment("world", 0.8, 0.2, 0.7)]


class FakeResult:
    def bestTranscription(self):
        return FakeTranscription()


def test_words_from_apple_result_are_absolute_word_tokens() -> None:
    words = _words_from_result(FakeResult(), _chunk())

    assert [(word.text, word.start_ts, word.end_ts, word.confidence) for word in words] == [
        ("hello", 12.25, 12.75, 0.9),
        ("world", 12.8, 13.0, 0.7),
    ]


class FakeRecognizerClass:
    authorized = 3
    requested = False

    @classmethod
    def authorizationStatus(cls):
        return cls.authorized

    @classmethod
    def requestAuthorization_(cls, handler):
        cls.requested = True
        handler(cls.authorized)


def test_apple_speech_transcriber_checks_authorization_on_macos(monkeypatch) -> None:
    from tidemark.transcribe import apple as apple_module

    modules = {
        "SFSpeechRecognizer": FakeRecognizerClass,
        "SFSpeechRecognizerAuthorizationStatusAuthorized": 3,
        "SFSpeechRecognizerAuthorizationStatusNotDetermined": 0,
    }
    monkeypatch.setattr("tidemark.transcribe.apple.platform.system", lambda: "Darwin")
    monkeypatch.setattr("tidemark.transcribe.apple._load_pyobjc_modules", lambda: modules)

    transcriber = AppleSpeechTranscriber()

    assert transcriber.language == "en-US"
    assert transcriber.engine == "apple-speech"
    assert transcriber._modules is modules
