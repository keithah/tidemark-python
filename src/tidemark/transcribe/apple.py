"""Apple Speech transcriber adapter for macOS."""

from __future__ import annotations

import platform
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from tidemark.audio import AudioChunk
from tidemark.transcribe.models import TranscriptResult, WordToken


class AppleSpeechUnavailable(RuntimeError):
    """Raised when Apple Speech cannot be used in the current runtime."""


class AppleSpeechTranscriptionError(RuntimeError):
    """Raised when Apple Speech fails to transcribe a chunk."""


class AppleSpeechTranscriber:
    """Transcribe audio chunks using macOS SFSpeechRecognizer via PyObjC."""

    language: str
    engine: str = "apple-speech"

    def __init__(self, *, language: str = "en-US", timeout_seconds: float = 60.0) -> None:
        self.language = language
        self.timeout_seconds = timeout_seconds
        if platform.system() != "Darwin":
            raise AppleSpeechUnavailable("apple speech is only available on macOS")
        self._modules = _load_pyobjc_modules()
        _ensure_speech_authorized(self._modules, timeout_seconds=timeout_seconds)

    def transcribe(self, chunk: AudioChunk) -> TranscriptResult:
        with tempfile.TemporaryDirectory(prefix="tidemark-speech-") as tmp_dir:
            wav_path = Path(tmp_dir) / f"segment-{chunk.segment_sequence}.wav"
            _write_pcm_wav(wav_path, chunk)
            return self._transcribe_wav(wav_path, chunk)

    def _transcribe_wav(self, wav_path: Path, chunk: AudioChunk) -> TranscriptResult:
        modules = self._modules
        locale = modules["NSLocale"].alloc().initWithLocaleIdentifier_(self.language)
        recognizer = modules["SFSpeechRecognizer"].alloc().initWithLocale_(locale)
        if recognizer is None or not bool(recognizer.isAvailable()):
            raise AppleSpeechUnavailable("apple speech recognizer is unavailable")

        request = modules["SFSpeechURLRecognitionRequest"].alloc().initWithURL_(modules["NSURL"].fileURLWithPath_(str(wav_path)))
        request.setShouldReportPartialResults_(False)
        if hasattr(request, "setAddsPunctuation_"):
            request.setAddsPunctuation_(True)

        done = threading.Event()
        result_box: dict[str, Any] = {}

        def handler(result: Any, error: Any) -> None:
            if error is not None:
                result_box["error"] = error
                done.set()
                return
            if result is not None and bool(result.isFinal()):
                result_box["result"] = result
                done.set()

        task = recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
        try:
            if not done.wait(self.timeout_seconds):
                raise AppleSpeechTranscriptionError("apple speech transcription timed out")
            if "error" in result_box:
                raise AppleSpeechTranscriptionError("apple speech transcription failed")
            result = result_box.get("result")
            if result is None:
                return TranscriptResult(words=(), language=self.language, engine=self.engine)
            words = _words_from_result(result, chunk)
            return TranscriptResult(words=tuple(words), language=self.language, engine=self.engine)
        finally:
            cancel = getattr(task, "cancel", None)
            if callable(cancel):
                cancel()


def _ensure_speech_authorized(modules: dict[str, Any], *, timeout_seconds: float) -> None:
    recognizer_class = modules["SFSpeechRecognizer"]
    authorized = modules["SFSpeechRecognizerAuthorizationStatusAuthorized"]
    not_determined = modules["SFSpeechRecognizerAuthorizationStatusNotDetermined"]
    status = recognizer_class.authorizationStatus()
    if status == authorized:
        return
    if status != not_determined:
        raise AppleSpeechUnavailable("apple speech is not authorized")

    done = threading.Event()
    result: dict[str, Any] = {}

    def handler(new_status: Any) -> None:
        result["status"] = new_status
        done.set()

    recognizer_class.requestAuthorization_(handler)
    if not done.wait(timeout_seconds):
        raise AppleSpeechUnavailable("apple speech authorization timed out")
    if result.get("status") != authorized:
        raise AppleSpeechUnavailable("apple speech is not authorized")


def _load_pyobjc_modules() -> dict[str, Any]:
    try:
        from Foundation import NSLocale, NSURL  # type: ignore
        from Speech import (  # type: ignore
            SFSpeechRecognizer,
            SFSpeechRecognizerAuthorizationStatusAuthorized,
            SFSpeechRecognizerAuthorizationStatusNotDetermined,
            SFSpeechURLRecognitionRequest,
        )
    except Exception as exc:  # pragma: no cover - depends on macOS/PyObjC availability.
        raise AppleSpeechUnavailable("apple speech PyObjC modules are unavailable") from exc
    return {
        "NSLocale": NSLocale,
        "NSURL": NSURL,
        "SFSpeechRecognizer": SFSpeechRecognizer,
        "SFSpeechRecognizerAuthorizationStatusAuthorized": SFSpeechRecognizerAuthorizationStatusAuthorized,
        "SFSpeechRecognizerAuthorizationStatusNotDetermined": SFSpeechRecognizerAuthorizationStatusNotDetermined,
        "SFSpeechURLRecognitionRequest": SFSpeechURLRecognitionRequest,
    }


def _write_pcm_wav(path: Path, chunk: AudioChunk) -> None:
    if chunk.sample_format != "s16le":
        raise AppleSpeechTranscriptionError("apple speech requires s16le PCM input")
    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(chunk.channels)
            handle.setsampwidth(2)
            handle.setframerate(chunk.sample_rate)
            handle.writeframes(chunk.pcm_bytes)
    except Exception as exc:
        raise AppleSpeechTranscriptionError("apple speech temporary WAV write failed") from exc


def _words_from_result(result: Any, chunk: AudioChunk) -> list[WordToken]:
    transcription = result.bestTranscription()
    segments = list(transcription.segments()) if transcription is not None else []
    words: list[WordToken] = []
    for segment in segments:
        text = str(segment.substring()).strip()
        if not text:
            continue
        start = chunk.start_ts + float(segment.timestamp())
        duration = float(segment.duration())
        confidence = float(segment.confidence()) if hasattr(segment, "confidence") else None
        words.append(WordToken(text=text, start_ts=start, end_ts=start + max(duration, 0.0), confidence=confidence))
    return words


__all__ = ["AppleSpeechTranscriber", "AppleSpeechTranscriptionError", "AppleSpeechUnavailable"]
