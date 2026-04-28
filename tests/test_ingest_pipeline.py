from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from collections.abc import Iterable
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.ingest.pipeline import (
    IngestPipelineProgress,
    IngestPipelineResult,
    TranscriptFixtureError,
    ingest_source_to_db,
    load_fixture_transcript,
)
from tidemark.fingerprint import AcoustIDLookupError, AcoustIDLookupResult, AudioFingerprint
from tidemark.ingest.segments import SegmentIngestError
from tidemark.search import TranscriptDatabaseEmpty, search_transcript_db
from tidemark.transcribe import DeterministicTranscriber


def _make_tiny_wav(path: Path) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.20:sample_rate=8000",
            "-ac",
            "1",
            "-y",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return path


def _write_manifest(path: Path, segment_name: str, *, include_cue: bool = True) -> Path:
    lines = [
        "#EXTM3U",
        "#EXT-X-MEDIA-SEQUENCE:37",
    ]
    if include_cue:
        lines.append("#EXT-X-CUE-OUT:DURATION=15.0")
    lines.extend(
        [
            "#EXTINF:0.20,",
            segment_name,
            "#EXT-X-CUE-IN",
            "#EXT-X-ENDLIST",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fetch_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _fingerprint_backend(sample_rate: int, channels: int, pcmiter: Iterable[bytes]) -> str:
    payload = b"".join(pcmiter)
    return f"fp:{sample_rate}:{channels}:{hashlib.sha256(payload).hexdigest()}"


def _secret_fingerprint_backend(_sample_rate: int, _channels: int, _pcmiter: Iterable[bytes]) -> str:
    return "RAW-FINGERPRINT-SECRET"


def _lookup_adapter(
    fingerprint: AudioFingerprint,
    *,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
) -> AcoustIDLookupResult:
    assert api_key in (None, "sk_live_secret_key")
    assert timeout_seconds in (None, 2.5)
    return AcoustIDLookupResult(
        acoustid_id=f"acoustid-{fingerprint.segment_sequence}",
        recording_id=f"recording-{fingerprint.segment_sequence}",
        title="Fixture tone",
        artist="Tidemark tests",
        album="Generated media",
        score=0.91,
        raw_status="ok",
        lookup_source="deterministic-test-backend",
    )


class FailingTranscriber:
    def transcribe(self, _chunk: object) -> object:
        raise RuntimeError("private transcript failure token=secret private phrase")


def test_ingest_source_to_db_persists_segments_words_markers_and_searches(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber(
            [
                ("hello", 0.03, 0.06, 0.9),
                ("tidemark", 0.07, 0.11, 0.8),
                ("search", 0.12, 0.16, None),
            ],
            language="en",
            engine="deterministic-fixture",
        ),
        source_url="fixture://integration/source.m3u8?token=secret",
    )

    assert isinstance(result, IngestPipelineResult)
    assert result.issues == ()
    assert len(result.segment_ids) == 1
    assert len(result.transcript_word_ids) == 3
    assert len(result.ad_event_ids) >= 1

    with sqlite3.connect(db_path) as conn:
        segment_rows = conn.execute(
            "SELECT id, sequence, start_ts, duration_seconds FROM segments ORDER BY id"
        ).fetchall()
        word_rows = conn.execute(
            "SELECT id, segment_id, segment_sequence, word_index, word_text, start_ts, end_ts, confidence "
            "FROM transcript_words ORDER BY id"
        ).fetchall()
        ad_rows = conn.execute(
            "SELECT id, marker_type, classification, source, tag, segment_seq, ts FROM ad_events ORDER BY id"
        ).fetchall()

    assert segment_rows == [(result.segment_ids[0], 37, 0.0, 0.2)]
    assert [row[0] for row in word_rows] == list(result.transcript_word_ids)
    assert [(row[1], row[2], row[3], row[4]) for row in word_rows] == [
        (result.segment_ids[0], 37, 0, "hello"),
        (result.segment_ids[0], 37, 1, "tidemark"),
        (result.segment_ids[0], 37, 2, "search"),
    ]
    assert [(row[5], row[6], row[7]) for row in word_rows] == [
        (pytest.approx(0.03), pytest.approx(0.06), 0.9),
        (pytest.approx(0.07), pytest.approx(0.11), 0.8),
        (pytest.approx(0.12), pytest.approx(0.16), None),
    ]
    assert any(row[2] == "AD_START" and row[4] == "#EXT-X-CUE-OUT" and row[5] == 37 for row in ad_rows)

    search_results = search_transcript_db(db_path, "tidemark search", context_seconds=1)
    assert len(search_results) == 1
    search_result = search_results[0]
    assert search_result.hit_start_ts == pytest.approx(0.07)
    assert search_result.hit_end_ts == pytest.approx(0.16)
    assert search_result.context_text == "hello tidemark search"
    assert search_result.word_ids == result.transcript_word_ids[1:]


def test_ingest_source_to_db_reports_progress_with_compact_redacted_counters(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"
    events: list[IngestPipelineProgress] = []

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber(
            [
                ("private transcript phrase", 0.03, 0.06, 0.9),
                ("RAW-FINGERPRINT-SECRET", 0.07, 0.11, 0.8),
            ],
            language="en",
            engine="deterministic-fixture",
        ),
        source_url="https://example.test/private/live.m3u8?token=secret",
        fingerprint=True,
        fingerprint_backend=_secret_fingerprint_backend,
        lookup_adapter=_lookup_adapter,
        acoustid_api_key="sk_live_secret_key",
        progress_callback=events.append,
    )

    assert result.issues == ()
    assert [event.phase for event in events] == ["resolving", "running", "completed"]
    assert events[0].counters == {
        "segments": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "words": 0,
        "markers": 0,
        "issues": 0,
        "retained": 0,
        "songs": 0,
    }
    assert events[1].counters == {
        "segments": 1,
        "processed": 1,
        "skipped": 0,
        "failed": 0,
        "words": 2,
        "markers": 1,
        "issues": 0,
        "retained": 1,
        "songs": 1,
    }
    assert events[-1].counters == events[1].counters
    serialized_events = json.dumps(
        [{"phase": event.phase, "counters": event.counters, "error": event.error} for event in events],
        sort_keys=True,
    )
    for secret in (
        "private transcript phrase",
        "RAW-FINGERPRINT-SECRET",
        "sk_live_secret_key",
        "token=secret",
        str(tmp_path),
    ):
        assert secret not in serialized_events


def test_restart_ingest_skips_duplicate_segment_without_downstream_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"
    source_url = "fixture://integration/source.m3u8?token=secret"
    first_events: list[IngestPipelineProgress] = []

    first_result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        fingerprint=True,
        fingerprint_backend=_fingerprint_backend,
        lookup_adapter=_lookup_adapter,
        source_url=source_url,
        progress_callback=first_events.append,
    )

    assert len(first_result.segment_ids) == 1
    assert first_result.skipped_segment_ids == ()
    assert first_result.issues == ()
    assert first_events[-1].counters == {
        "segments": 1,
        "processed": 1,
        "skipped": 0,
        "failed": 0,
        "words": 1,
        "markers": 1,
        "issues": 0,
        "retained": 1,
        "songs": 1,
    }
    counts_after_first = {table: _fetch_count(db_path, table) for table in _timeline_tables()}

    def fail_decode(_segment: object) -> object:
        raise AssertionError("duplicate skip path must not decode audio")

    def fail_fingerprint_backend(_sample_rate: int, _channels: int, _pcmiter: Iterable[bytes]) -> str:
        raise AssertionError("duplicate skip path must not fingerprint audio")

    def fail_lookup_adapter(
        _fingerprint: AudioFingerprint,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AcoustIDLookupResult:
        raise AssertionError("duplicate skip path must not lookup fingerprints")

    class FailTranscriber:
        def transcribe(self, _chunk: object) -> object:
            raise AssertionError("duplicate skip path must not transcribe audio")

    monkeypatch.setattr("tidemark.audio.decode_segment_audio", fail_decode)
    restart_events: list[IngestPipelineProgress] = []

    restart_result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=FailTranscriber(),  # type: ignore[arg-type]
        fingerprint=True,
        fingerprint_backend=fail_fingerprint_backend,
        lookup_adapter=fail_lookup_adapter,
        source_url=source_url,
        progress_callback=restart_events.append,
    )

    assert restart_result.segment_ids == ()
    assert restart_result.skipped_segment_ids == first_result.segment_ids
    assert restart_result.transcript_word_ids == ()
    assert restart_result.ad_event_ids == ()
    assert restart_result.retained_audio_ids == ()
    assert restart_result.song_ids == ()
    assert restart_result.issues == ()
    assert {table: _fetch_count(db_path, table) for table in _timeline_tables()} == counts_after_first
    assert restart_events[-1].counters == {
        "segments": 1,
        "processed": 0,
        "skipped": 1,
        "failed": 0,
        "words": 0,
        "markers": 0,
        "issues": 0,
        "retained": 0,
        "songs": 0,
    }


def test_restart_ingest_processes_sha_mismatch_as_new_work(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"
    source_url = "fixture://integration/source.m3u8"

    first_result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        source_url=source_url,
        include_manifest_markers=False,
    )
    _make_tiny_wav(media_path)
    media_path.write_bytes(media_path.read_bytes() + b"changed")

    second_result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        source_url=source_url,
        include_manifest_markers=False,
    )

    assert len(first_result.segment_ids) == 1
    assert len(second_result.segment_ids) == 1
    assert second_result.skipped_segment_ids == ()
    assert _fetch_count(db_path, "segments") == 2
    assert _fetch_count(db_path, "transcript_words") == 2


def test_restart_ingest_processes_source_mismatch_as_new_work(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        source_url="fixture://integration/source-a.m3u8",
        include_manifest_markers=False,
    )
    second_result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        source_url="fixture://integration/source-b.m3u8",
        include_manifest_markers=False,
    )

    assert len(second_result.segment_ids) == 1
    assert second_result.skipped_segment_ids == ()
    assert _fetch_count(db_path, "segments") == 2
    assert _fetch_count(db_path, "transcript_words") == 2


def test_restart_evidence_lookup_failure_aborts_with_redacted_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "private-playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    def fail_lookup(**_kwargs: object) -> object:
        raise RuntimeError("sqlite exploded /private/path token=secret raw fingerprint")

    monkeypatch.setattr("tidemark.store.find_segment_by_restart_evidence", fail_lookup)

    with pytest.raises(RuntimeError, match="pipeline segment restart lookup failed") as exc_info:
        ingest_source_to_db(
            manifest,
            db_path=db_path,
            transcriber=DeterministicTranscriber([("private phrase", 0.0, 0.1, None)]),
            source_url="https://example.test/private/live.m3u8?token=secret",
            include_manifest_markers=False,
        )

    error_message = str(exc_info.value)
    for secret in ("token=secret", "private phrase", "/private/path", "raw fingerprint"):
        assert secret not in error_message
    assert _fetch_count(db_path, "segments") == 0



def _timeline_tables() -> tuple[str, ...]:
    return ("segments", "transcript_words", "ad_events", "retained_audio", "songs")



def test_ingest_progress_callback_failures_do_not_change_pipeline_result(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"
    attempts = 0

    def failing_callback(_progress: IngestPipelineProgress) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("reporter write failed token=secret")

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
        progress_callback=failing_callback,
    )

    assert attempts >= 2
    assert result.issues == ()
    assert len(result.segment_ids) == 1
    assert len(result.transcript_word_ids) == 1
    assert _fetch_count(db_path, "segments") == 1
    assert _fetch_count(db_path, "transcript_words") == 1


def test_default_ingest_does_not_create_fingerprint_or_retained_audio_side_effects(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
    )

    assert result.issues == ()
    assert len(result.segment_ids) == 1
    assert len(result.transcript_word_ids) == 1
    assert not (tmp_path / "tidemark-audio").exists()
    assert _fetch_count(db_path, "songs") == 0
    assert _fetch_count(db_path, "fingerprint_cache") == 0
    assert _fetch_count(db_path, "retained_audio") == 0


def test_fingerprint_ingest_without_transcriber_writes_retained_audio_cache_song_and_markers(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=None,
        fingerprint=True,
        fingerprint_backend=_fingerprint_backend,
        lookup_adapter=_lookup_adapter,
        acoustid_api_key="sk_live_secret_key",
        lookup_timeout_seconds=2.5,
        source_url="fixture://integration/source.m3u8?token=secret",
    )

    assert result.issues == ()
    assert len(result.segment_ids) == 1
    assert result.transcript_word_ids == ()
    assert len(result.ad_event_ids) >= 1
    assert len(result.retained_audio_ids) == 1
    assert len(result.song_ids) == 1
    assert _fetch_count(db_path, "segments") == 1
    assert _fetch_count(db_path, "transcript_words") == 0
    assert _fetch_count(db_path, "ad_events") >= 1
    assert _fetch_count(db_path, "retained_audio") == 1
    assert _fetch_count(db_path, "fingerprint_cache") == 1
    assert _fetch_count(db_path, "songs") == 1

    with sqlite3.connect(db_path) as conn:
        retained_path = Path(conn.execute("SELECT path FROM retained_audio").fetchone()[0])
        song_row = conn.execute("SELECT id, title, lookup_source FROM songs").fetchone()
        cache_row = conn.execute("SELECT title, lookup_source FROM fingerprint_cache").fetchone()

    assert retained_path.parent == tmp_path / "tidemark-audio"
    assert retained_path.exists()
    assert retained_path.name.startswith("segment-37-")
    assert media_path.stem not in retained_path.name
    assert song_row == (result.song_ids[0], "Fixture tone", "deterministic-test-backend")
    assert cache_row == ("Fixture tone", "deterministic-test-backend")


def test_fingerprint_ingest_with_transcriber_writes_words_and_fingerprint_evidence(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        fingerprint=True,
        fingerprint_backend=_fingerprint_backend,
        lookup_adapter=_lookup_adapter,
        include_manifest_markers=False,
    )

    assert result.issues == ()
    assert len(result.transcript_word_ids) == 1
    assert len(result.retained_audio_ids) == 1
    assert len(result.song_ids) == 1
    assert _fetch_count(db_path, "transcript_words") == 1
    assert _fetch_count(db_path, "retained_audio") == 1
    assert _fetch_count(db_path, "fingerprint_cache") == 1
    assert _fetch_count(db_path, "songs") == 1


def test_transcription_failure_still_permits_retention_fingerprint_and_lookup(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=FailingTranscriber(),  # type: ignore[arg-type]
        fingerprint=True,
        fingerprint_backend=_fingerprint_backend,
        lookup_adapter=_lookup_adapter,
        source_url="fixture://integration/private.m3u8?token=secret",
        include_manifest_markers=False,
    )

    assert [(issue.phase, issue.segment_sequence, issue.message) for issue in result.issues] == [
        ("transcribe", 37, "transcription failed")
    ]
    assert result.transcript_word_ids == ()
    assert len(result.retained_audio_ids) == 1
    assert len(result.song_ids) == 1
    assert _fetch_count(db_path, "transcript_words") == 0
    assert _fetch_count(db_path, "retained_audio") == 1
    assert _fetch_count(db_path, "fingerprint_cache") == 1
    assert _fetch_count(db_path, "songs") == 1
    for issue in result.issues:
        assert "token=secret" not in issue.message
        assert "private phrase" not in issue.message


def test_retention_failure_records_issue_but_still_permits_fingerprint_and_lookup(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("blocks mkdir", encoding="utf-8")

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=None,
        fingerprint=True,
        fingerprint_backend=_fingerprint_backend,
        lookup_adapter=_lookup_adapter,
        retention_dir=not_a_directory / "child",
        include_manifest_markers=False,
    )

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.phase == "retain_audio"
    assert issue.segment_sequence == 37
    assert "Retention error" in issue.message or "retention failed" in issue.message
    assert result.retained_audio_ids == ()
    assert len(result.song_ids) == 1
    assert _fetch_count(db_path, "retained_audio") == 0
    assert _fetch_count(db_path, "fingerprint_cache") == 1
    assert _fetch_count(db_path, "songs") == 1


def test_fingerprint_failure_records_redacted_issue_without_lookup_or_raw_evidence(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    def exploding_backend(_sample_rate: int, _channels: int, _pcmiter: Iterable[bytes]) -> str:
        raise RuntimeError("private backend exploded RAW-FINGERPRINT-SECRET token=secret")

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=None,
        fingerprint=True,
        fingerprint_backend=exploding_backend,
        lookup_adapter=_lookup_adapter,
        source_url="fixture://integration/private.m3u8?token=secret",
        include_manifest_markers=False,
    )

    assert len(result.retained_audio_ids) == 1
    assert result.song_ids == ()
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.phase == "fingerprint"
    assert issue.segment_sequence == 37
    assert issue.message == "fingerprint failed"
    assert _fetch_count(db_path, "fingerprint_cache") == 0
    assert _fetch_count(db_path, "songs") == 0
    for secret in ("RAW-FINGERPRINT-SECRET", "token=secret", "private backend exploded"):
        assert secret not in issue.message


def test_lookup_failure_records_redacted_issue_without_raw_fingerprint_or_secret_text(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    def failing_lookup(
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

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=None,
        fingerprint=True,
        fingerprint_backend=_secret_fingerprint_backend,
        lookup_adapter=failing_lookup,
        acoustid_api_key="sk_live_secret_key",
        source_url="fixture://integration/private.m3u8?token=secret",
        include_manifest_markers=False,
    )

    assert len(result.retained_audio_ids) == 1
    assert result.song_ids == ()
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.phase == "lookup"
    assert issue.segment_sequence == 37
    assert "timeout" in issue.message
    assert _fetch_count(db_path, "fingerprint_cache") == 0
    assert _fetch_count(db_path, "songs") == 0
    for secret in (
        "RAW-FINGERPRINT-SECRET",
        "sk_live_secret_key",
        "token=secret",
        "private backend exploded",
    ):
        assert secret not in issue.message


def test_load_fixture_transcript_accepts_json_array(tmp_path: Path) -> None:
    fixture_path = tmp_path / "transcript.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"text": "hello", "start_offset": 0.0, "end_offset": 0.1, "confidence": 0.75},
                {"text": "tidemark", "start_offset": 0.2, "end_offset": 0.3},
            ]
        ),
        encoding="utf-8",
    )

    assert load_fixture_transcript(fixture_path) == (
        ("hello", 0.0, 0.1, 0.75),
        ("tidemark", 0.2, 0.3, None),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "json"),
        ({"text": "hello"}, "array"),
        ([{"text": "", "start_offset": 0.0, "end_offset": 0.1}], "text"),
        ([{"text": "hello", "start_offset": -0.1, "end_offset": 0.1}], "start_offset"),
        ([{"text": "hello", "start_offset": 0.2, "end_offset": 0.1}], "end_offset"),
        ([{"text": "hello", "start_offset": 0.0, "end_offset": 0.1, "confidence": 1.1}], "confidence"),
    ],
)
def test_load_fixture_transcript_rejects_malformed_field_only(tmp_path: Path, payload: object, message: str) -> None:
    fixture_path = tmp_path / "transcript.json"
    if isinstance(payload, str):
        fixture_path.write_text(payload, encoding="utf-8")
    else:
        fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TranscriptFixtureError, match=message) as exc_info:
        load_fixture_transcript(fixture_path)

    error_message = str(exc_info.value)
    assert "hello" not in error_message
    assert str(fixture_path) not in error_message


def test_network_source_url_is_rejected_before_db_work(tmp_path: Path) -> None:
    db_path = tmp_path / "tidemark.sqlite3"

    with pytest.raises(SegmentIngestError, match="network URL"):
        ingest_source_to_db(
            "https://example.test/private.m3u8?token=secret",
            db_path=db_path,
            transcriber=DeterministicTranscriber([]),
        )

    assert not db_path.exists()


def test_invalid_media_preserves_segment_row_and_returns_redacted_decode_issue(tmp_path: Path) -> None:
    bad_media = tmp_path / "private-bad-segment.wav"
    bad_media.write_bytes(b"not a wav segment")
    manifest = _write_manifest(tmp_path / "private-playlist.m3u8", bad_media.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("private phrase", 0.0, 0.1, None)]),
        source_url="fixture://integration/private.m3u8?token=secret",
    )

    assert len(result.segment_ids) == 1
    assert result.transcript_word_ids == ()
    assert len(result.ad_event_ids) >= 1
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.phase == "decode"
    assert issue.segment_sequence == 37
    assert "ffmpeg" in issue.message
    assert "private phrase" not in issue.message
    assert "token=secret" not in issue.message
    assert "private" not in issue.message
    assert _fetch_count(db_path, "segments") == 1
    assert _fetch_count(db_path, "transcript_words") == 0


def test_include_manifest_markers_false_writes_no_ad_events(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.sqlite3"

    result = ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
    )

    assert result.ad_event_ids == ()
    assert _fetch_count(db_path, "ad_events") == 0


def test_no_match_search_returns_empty_tuple(tmp_path: Path) -> None:
    media_path = _make_tiny_wav(tmp_path / "segment37.wav")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", media_path.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([("hello", 0.0, 0.1, None)]),
        include_manifest_markers=False,
    )

    assert search_transcript_db(db_path, "not present") == ()


def test_no_match_search_on_decode_only_db_has_no_transcripts(tmp_path: Path) -> None:
    bad_media = tmp_path / "bad-segment.wav"
    bad_media.write_bytes(b"not media")
    manifest = _write_manifest(tmp_path / "playlist.m3u8", bad_media.name, include_cue=False)
    db_path = tmp_path / "tidemark.sqlite3"

    ingest_source_to_db(
        manifest,
        db_path=db_path,
        transcriber=DeterministicTranscriber([]),
        include_manifest_markers=False,
    )

    with pytest.raises(TranscriptDatabaseEmpty, match="transcript_words"):
        search_transcript_db(db_path, "anything")
