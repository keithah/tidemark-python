from __future__ import annotations

import sqlite3
from pathlib import Path

from tidemark.ingest.pipeline import ingest_live_hls_to_db
from tidemark.ingest.segments import iter_live_hls_segments
from tidemark.transcribe import WordToken
from tidemark.transcribe.models import TranscriptResult


class FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class OneWordTranscriber:
    def transcribe(self, chunk):
        return TranscriptResult(words=(WordToken("hello", chunk.start_ts, chunk.start_ts + 0.1, 0.5),), engine="test")


def test_iter_live_hls_segments_resolves_network_segments_once(monkeypatch) -> None:
    manifest = """#EXTM3U
#EXT-X-MEDIA-SEQUENCE:42
#EXTINF:0.2,
segment42.ts
#EXT-X-ENDLIST
"""
    opened: list[str] = []

    def fake_load(source, *, timeout=None, headers=None, response=None):
        assert source == "https://cdn.example/live/playlist.m3u8"
        return manifest

    def fake_open(url, *, timeout=None, headers=None):
        opened.append(url)
        return FakeHttpResponse(b"segment bytes")

    monkeypatch.setattr("tidemark.monitor_sources._load_hls_manifest_text", fake_load)
    monkeypatch.setattr("tidemark.monitor_sources._open_http_response", fake_open)

    segments = list(iter_live_hls_segments("https://cdn.example/live/playlist.m3u8", timeout=1.0))

    assert len(segments) == 1
    assert segments[0].sequence == 42
    assert segments[0].resolved_uri == "https://cdn.example/live/segment42.ts"
    assert segments[0].load_bytes() == b"segment bytes"
    assert opened == ["https://cdn.example/live/segment42.ts"]


def test_ingest_live_hls_to_db_decodes_transcribes_and_stores_words(monkeypatch, tmp_path: Path) -> None:
    from tidemark.audio import AudioChunk

    segment_payload = b"not real media"
    manifest = """#EXTM3U
#EXT-X-MEDIA-SEQUENCE:7
#EXTINF:0.2,
segment7.ts
#EXT-X-ENDLIST
"""

    def fake_load(source, *, timeout=None, headers=None, response=None):
        return manifest

    def fake_open(url, *, timeout=None, headers=None):
        return FakeHttpResponse(segment_payload)

    def fake_decode(segment):
        return AudioChunk(
            pcm_bytes=b"\x00\x00" * 100,
            sample_rate=16000,
            channels=1,
            sample_format="s16le",
            segment_sequence=segment.sequence,
            source_url=segment.source_url,
            resolved_uri=segment.resolved_uri,
            start_ts=segment.start_ts,
            duration_seconds=segment.duration_seconds,
            byte_length=200,
            metadata={},
        )

    monkeypatch.setattr("tidemark.monitor_sources._load_hls_manifest_text", fake_load)
    monkeypatch.setattr("tidemark.monitor_sources._open_http_response", fake_open)
    monkeypatch.setattr("tidemark.audio.decode_segment_audio", fake_decode)

    db_path = tmp_path / "live.sqlite"
    result = ingest_live_hls_to_db(
        "https://cdn.example/live/playlist.m3u8",
        db_path=db_path,
        transcriber=OneWordTranscriber(),
        timeout=1.0,
    )

    assert len(result.segment_ids) == 1
    assert len(result.transcript_word_ids) == 1
    assert result.issues == ()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("select word_text from transcript_words").fetchone()[0] == "hello"
