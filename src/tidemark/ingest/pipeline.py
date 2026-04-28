"""Library-first ingest pipeline composition for deterministic local sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tidemark.ingest.segments import SegmentRecord, resolve_segments

if TYPE_CHECKING:
    from tidemark.transcribe import Transcriber

FixtureTranscriptWord = tuple[str, float, float, float | None]


@dataclass(frozen=True)
class IngestIssue:
    """Redacted per-segment ingest issue."""

    phase: str
    segment_sequence: int | None
    message: str


@dataclass(frozen=True)
class IngestPipelineProgress:
    """Compact, redacted ingest progress counters for runtime health surfaces."""

    phase: str
    counters: dict[str, int]
    error: str | None = None


ProgressCallback = Callable[[IngestPipelineProgress], None]


@dataclass(frozen=True)
class IngestPipelineResult:
    """Counters and issue records returned by the deterministic ingest pipeline."""

    segment_ids: tuple[int, ...]
    transcript_word_ids: tuple[int, ...]
    ad_event_ids: tuple[int, ...]
    issues: tuple[IngestIssue, ...]
    retained_audio_ids: tuple[int, ...] = ()
    song_ids: tuple[int, ...] = ()


class TranscriptFixtureError(ValueError):
    """Fixture transcript validation error with field-only diagnostics."""


def load_fixture_transcript(path: str | Path) -> tuple[FixtureTranscriptWord, ...]:
    """Load deterministic transcript words from a JSON fixture file."""
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        raise TranscriptFixtureError("fixture file could not be read") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TranscriptFixtureError("json must be valid") from exc

    if not isinstance(payload, list):
        raise TranscriptFixtureError("root must be a json array")

    return tuple(_fixture_word_from_object(item) for item in payload)


def ingest_source_to_db(
    source: str | Path,
    *,
    db_path: str | Path,
    transcriber: Transcriber | None,
    source_url: str | None = None,
    include_manifest_markers: bool = True,
    fingerprint: bool = False,
    fingerprint_backend: Any = None,
    lookup_adapter: Any = None,
    acoustid_api_key: str | None = None,
    lookup_timeout_seconds: float | None = None,
    retention_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IngestPipelineResult:
    """Resolve, store, decode, and optionally transcribe/fingerprint local media input.

    The pipeline is intentionally local-only: source resolution is delegated to
    ``resolve_segments()``, which rejects network URLs with redacted diagnostics.
    Optional branches are independent so transcription, retention, fingerprint,
    and lookup failures surface as redacted per-segment issues without blocking
    later optional work for the same decoded segment.
    """
    _notify_progress(progress_callback, "resolving", _empty_progress_counters())
    segments = resolve_segments(source, source_url=source_url)
    from tidemark.store import initialize_db, insert_transcript_words

    conn = initialize_db(db_path)
    try:
        segment_ids: list[int] = []
        transcript_word_ids: list[int] = []
        retained_audio_ids: list[int] = []
        song_ids: list[int] = []
        ad_event_ids = _insert_manifest_markers(
            conn,
            source,
            source_url=source_url,
            include_manifest_markers=include_manifest_markers,
        )
        issues: list[IngestIssue] = []

        for segment in segments:
            segment_id = _insert_segment(conn, segment)
            segment_ids.append(segment_id)

            try:
                from tidemark.audio import AudioDecodeError, decode_segment_audio

                chunk = decode_segment_audio(segment)
            except AudioDecodeError as exc:
                issues.append(_issue("decode", segment.sequence, str(exc)))
                continue
            except Exception:
                issues.append(_issue("decode", segment.sequence, "audio decode failed"))
                continue

            if transcriber is not None:
                try:
                    transcript = transcriber.transcribe(chunk)
                except Exception:
                    issues.append(_issue("transcribe", segment.sequence, "transcription failed"))
                else:
                    try:
                        transcript_word_ids.extend(
                            insert_transcript_words(
                                conn,
                                segment_id=segment_id,
                                source_url=chunk.source_url,
                                segment_sequence=chunk.segment_sequence,
                                words=transcript.words,
                            )
                        )
                    except Exception as exc:
                        issues.append(_issue("store_transcript", segment.sequence, _safe_store_message(exc)))

            if fingerprint:
                _run_fingerprint_branches(
                    conn,
                    db_path=db_path,
                    chunk=chunk,
                    segment_id=segment_id,
                    fingerprint_backend=fingerprint_backend,
                    lookup_adapter=lookup_adapter,
                    acoustid_api_key=acoustid_api_key,
                    lookup_timeout_seconds=lookup_timeout_seconds,
                    retention_dir=retention_dir,
                    retained_audio_ids=retained_audio_ids,
                    song_ids=song_ids,
                    issues=issues,
                )

            _notify_progress(
                progress_callback,
                "running",
                _progress_counters(
                    segment_ids=segment_ids,
                    transcript_word_ids=transcript_word_ids,
                    ad_event_ids=ad_event_ids,
                    issues=issues,
                    retained_audio_ids=retained_audio_ids,
                    song_ids=song_ids,
                ),
            )

        result = IngestPipelineResult(
            segment_ids=tuple(segment_ids),
            transcript_word_ids=tuple(transcript_word_ids),
            ad_event_ids=tuple(ad_event_ids),
            issues=tuple(issues),
            retained_audio_ids=tuple(retained_audio_ids),
            song_ids=tuple(song_ids),
        )
        _notify_progress(progress_callback, "completed", _result_counters(result))
        return result
    finally:
        conn.close()


def _empty_progress_counters() -> dict[str, int]:
    return {"segments": 0, "words": 0, "markers": 0, "issues": 0, "retained": 0, "songs": 0}


def _progress_counters(
    *,
    segment_ids: list[int],
    transcript_word_ids: list[int],
    ad_event_ids: list[int],
    issues: list[IngestIssue],
    retained_audio_ids: list[int],
    song_ids: list[int],
) -> dict[str, int]:
    return {
        "segments": len(segment_ids),
        "words": len(transcript_word_ids),
        "markers": len(ad_event_ids),
        "issues": len(issues),
        "retained": len(retained_audio_ids),
        "songs": len(song_ids),
    }


def _result_counters(result: IngestPipelineResult) -> dict[str, int]:
    return {
        "segments": len(result.segment_ids),
        "words": len(result.transcript_word_ids),
        "markers": len(result.ad_event_ids),
        "issues": len(result.issues),
        "retained": len(result.retained_audio_ids),
        "songs": len(result.song_ids),
    }


def _notify_progress(
    progress_callback: ProgressCallback | None,
    phase: str,
    counters: dict[str, int],
    *,
    error: str | None = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(IngestPipelineProgress(phase=phase, counters=dict(counters), error=error))
    except Exception:
        pass


def _fixture_word_from_object(item: object) -> FixtureTranscriptWord:
    if not isinstance(item, dict):
        raise TranscriptFixtureError("item must be an object")

    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TranscriptFixtureError("text must be a non-empty string")

    start_offset = _fixture_number(item.get("start_offset"), "start_offset")
    end_offset = _fixture_number(item.get("end_offset"), "end_offset")
    if start_offset < 0:
        raise TranscriptFixtureError("start_offset must be >= 0")
    if end_offset < start_offset:
        raise TranscriptFixtureError("end_offset must be >= start_offset")

    confidence_value = item.get("confidence")
    confidence: float | None
    if confidence_value is None:
        confidence = None
    else:
        confidence = _fixture_number(confidence_value, "confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise TranscriptFixtureError("confidence must be between 0 and 1")

    return (text, start_offset, end_offset, confidence)


def _fixture_number(value: object, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TranscriptFixtureError(f"{field} must be numeric")
    return float(value)


def _insert_segment(conn: Any, segment: SegmentRecord) -> int:
    try:
        from tidemark.store import insert_segment

        return insert_segment(
            conn,
            source_url=segment.source_url,
            sequence=segment.sequence,
            resolved_uri=segment.resolved_uri,
            local_path=segment.local_path,
            start_ts=segment.start_ts,
            duration_seconds=segment.duration_seconds if segment.duration_seconds is not None else 0.0,
            byte_length=segment.byte_length,
            sha256=segment.sha256,
            metadata=segment.metadata,
        )
    except Exception as exc:
        raise RuntimeError("pipeline segment store failed") from exc


def _insert_manifest_markers(
    conn: Any,
    source: str | Path,
    *,
    source_url: str | None,
    include_manifest_markers: bool,
) -> list[int]:
    if not include_manifest_markers:
        return []

    source_path = Path(source)
    if source_path.suffix.lower() != ".m3u8":
        return []

    try:
        manifest_text = source_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("pipeline manifest marker read failed") from exc

    manifest_url = source_url or source_path.resolve().as_uri()
    marker_ids: list[int] = []
    try:
        from tidemark.ingest.hls import iter_hls_manifest_scte35_markers
        from tidemark.markers.classifier import Classifier
        from tidemark.store import insert_ad_event

        classifier = Classifier()
        for marker in iter_hls_manifest_scte35_markers(
            manifest_text,
            manifest_url=manifest_url,
            timestamp=0.0,
        ):
            classifier.classify(marker)
            marker_ids.append(insert_ad_event(conn, manifest_url, marker))
    except Exception as exc:
        raise RuntimeError("pipeline manifest marker store failed") from exc
    return marker_ids


def _run_fingerprint_branches(
    conn: Any,
    *,
    db_path: str | Path,
    chunk: Any,
    segment_id: int,
    fingerprint_backend: Any,
    lookup_adapter: Any,
    acoustid_api_key: str | None,
    lookup_timeout_seconds: float | None,
    retention_dir: str | Path | None,
    retained_audio_ids: list[int],
    song_ids: list[int],
    issues: list[IngestIssue],
) -> None:
    sequence = chunk.segment_sequence

    try:
        from tidemark.fingerprint import RetentionError, write_retained_audio
        from tidemark.store import insert_retained_audio

        retained = write_retained_audio(chunk, db_path=db_path, retention_dir=retention_dir)
        try:
            retained_audio_ids.append(
                insert_retained_audio(
                    conn,
                    segment_id=segment_id,
                    source_url=chunk.source_url,
                    segment_sequence=chunk.segment_sequence,
                    path=str(retained.path),
                    format=retained.format,
                    sample_rate=retained.sample_rate,
                    channels=retained.channels,
                    sample_format=retained.sample_format,
                    start_ts=retained.start_ts,
                    duration_seconds=retained.duration_seconds,
                    byte_length=retained.byte_length,
                    sha256=retained.sha256,
                )
            )
        except Exception:
            issues.append(_issue("store_retained_audio", sequence, "retained audio store failed"))
    except RetentionError as exc:
        issues.append(_issue("retain_audio", sequence, str(exc)))
    except Exception:
        issues.append(_issue("retain_audio", sequence, "retention failed"))

    try:
        from tidemark.fingerprint import FingerprintError, fingerprint_audio_chunk

        fingerprint_result = fingerprint_audio_chunk(chunk, backend=fingerprint_backend)
    except FingerprintError:
        issues.append(_issue("fingerprint", sequence, "fingerprint failed"))
        return
    except Exception:
        issues.append(_issue("fingerprint", sequence, "fingerprint failed"))
        return

    try:
        from tidemark.fingerprint import AcoustIDLookupError, PyAcoustIDLookupAdapter, identify_fingerprint

        identification = identify_fingerprint(
            conn,
            fingerprint_result,
            segment_id,
            lookup_adapter or PyAcoustIDLookupAdapter(),
            api_key=acoustid_api_key,
            timeout_seconds=lookup_timeout_seconds,
        )
        song_ids.append(identification.song_id)
    except AcoustIDLookupError as exc:
        issues.append(_issue("lookup", sequence, _safe_lookup_message(exc)))
    except Exception:
        issues.append(_issue("lookup", sequence, "lookup failed"))


def _safe_lookup_message(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.strip():
        return f"lookup {status.strip()}"
    return "lookup failed"


def _issue(phase: str, segment_sequence: int | None, message: str) -> IngestIssue:
    return IngestIssue(phase=phase, segment_sequence=segment_sequence, message=message)


def _safe_store_message(exc: Exception) -> str:
    message = str(exc)
    if "insert_transcript_words()" in message:
        return message
    return "transcript word store failed"


__all__ = [
    "IngestIssue",
    "IngestPipelineProgress",
    "IngestPipelineResult",
    "TranscriptFixtureError",
    "ingest_source_to_db",
    "load_fixture_transcript",
]
