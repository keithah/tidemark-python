from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from tidemark.store import initialize_db, insert_segment, insert_transcript_words
from tidemark.transcribe import WordToken


CLI = Path(".venv/bin/tidemark")
FIXTURE = Path("tests/fixtures/scte35_splice_null.ts")
EXPECTED_MARKER_KEYS = [
    "Type",
    "Classification",
    "Source",
    "Tag",
    "PTS",
    "Segment",
    "RawBase64",
    "Command",
    "Descriptors",
    "Tags",
    "Fields",
    "Timestamp",
]


def run_tidemark(*args: object, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    assert CLI.exists(), "expected editable install to provide .venv/bin/tidemark"
    command = [str(CLI), *[str(arg) for arg in args]]
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def parse_ndjson(stdout: str) -> list[dict[str, object]]:
    assert "Traceback" not in stdout
    if not stdout:
        return []
    return [json.loads(line) for line in stdout.splitlines()]


def read_ad_event_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return list(
            conn.execute(
                """
                select source_url, marker_type, classification, source, tag,
                       segment_seq, pts, break_duration, raw_json, ts
                from ad_events order by id
                """
            )
        )
    finally:
        conn.close()


def comparable_marker(marker: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in marker.items() if key != "Timestamp"}


def assert_redacted_diagnostics(stderr: str) -> None:
    assert "Traceback" not in stderr
    assert "token=" not in stderr
    assert "raw_base64" not in stderr
    assert "RawBase64" not in stderr
    assert "/DAR" not in stderr


def create_search_fixture_db(db_path: Path) -> tuple[int, tuple[int, ...]]:
    conn = initialize_db(db_path)
    try:
        segment_id = insert_segment(
            conn,
            source_url="https://example.test/live/channel.m3u8?token=secret",
            sequence=3,
            resolved_uri="https://cdn.example.test/media/segment-3.ts",
            local_path=None,
            start_ts=10.0,
            duration_seconds=6.0,
            byte_length=2048,
            sha256="b" * 64,
        )
        word_ids = insert_transcript_words(
            conn,
            segment_id=segment_id,
            source_url="https://example.test/live/channel.m3u8?token=secret",
            segment_sequence=3,
            words=(
                WordToken(text="hello", start_ts=10.0, end_ts=10.4, confidence=0.95),
                WordToken(text="tidemark", start_ts=10.5, end_ts=10.9, confidence=0.96),
                WordToken(text="search", start_ts=11.0, end_ts=11.4, confidence=0.97),
            ),
        )
        return segment_id, word_ids
    finally:
        conn.close()


def test_installed_search_json_smoke_reads_transcript_words_from_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "transcripts.sqlite"
    segment_id, word_ids = create_search_fixture_db(db_path)

    result = run_tidemark("search", "tidemark", "--db", db_path, "--context", "1", "--json")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert rows[0] == {
        "source_url": "https://example.test/live/channel.m3u8?token=secret",
        "segment_id": segment_id,
        "segment_sequence": 3,
        "hit_start_ts": 10.5,
        "hit_end_ts": 10.9,
        "context_start_ts": 10.0,
        "context_end_ts": 11.4,
        "context_text": "hello tidemark search",
        "matched_text": "tidemark",
        "word_ids": [word_ids[1]],
    }


def test_installed_search_json_no_match_returns_empty_array(tmp_path: Path) -> None:
    db_path = tmp_path / "transcripts.sqlite"
    create_search_fixture_db(db_path)

    result = run_tidemark("search", "absent", "--db", db_path, "--json")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == []


def test_installed_search_missing_db_is_redacted_and_does_not_create_file(tmp_path: Path) -> None:
    missing_db = tmp_path / "private-missing.sqlite"

    result = run_tidemark("search", "private-query", "--db", missing_db, "--json")

    assert result.returncode == 1
    assert result.stdout == ""
    assert not missing_db.exists()
    assert "[tidemark] error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert "private-query" not in result.stderr
    assert "hello" not in result.stderr
    assert str(missing_db) not in result.stderr
    assert missing_db.name not in result.stderr


def test_installed_monitor_fixture_emits_go_compatible_scte35_ndjson() -> None:
    result = run_tidemark("monitor", FIXTURE, "--stream-type", "mpegts", "--json", "--timeout", "1")

    assert result.returncode == 0, result.stderr
    markers = parse_ndjson(result.stdout)
    assert markers, "expected tracked MPEGTS fixture to emit at least one marker"
    first = markers[0]
    assert list(first) == EXPECTED_MARKER_KEYS
    assert first["Type"] == "SCTE35"
    assert first["Source"] == "mpegts"
    assert first["Classification"] == "UNKNOWN"
    assert first["Fields"] == {"CommandName": "Splice Null"}
    assert result.stderr.startswith("[tidemark] completed: reason=")
    assert_redacted_diagnostics(result.stderr)


def test_json_out_and_sqlite_raw_json_match_installed_cli_stdout(tmp_path: Path) -> None:
    json_out = tmp_path / "markers.ndjson"
    db_path = tmp_path / "events.sqlite"

    result = run_tidemark(
        "monitor",
        FIXTURE,
        "--stream-type",
        "mpegts",
        "--json",
        "--json-out",
        json_out,
        "--db",
        db_path,
        "--timeout",
        "1",
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = result.stdout.splitlines()
    assert stdout_lines, "expected stdout NDJSON markers"
    assert json_out.read_text(encoding="utf-8").splitlines() == stdout_lines

    rows = read_ad_event_rows(db_path)
    assert [row["raw_json"] for row in rows] == stdout_lines
    assert len(rows) == len(stdout_lines)
    for row, line in zip(rows, stdout_lines, strict=True):
        marker = json.loads(line)
        assert row["source_url"] == str(FIXTURE)
        assert row["marker_type"] == marker["Type"] == "SCTE35"
        assert row["classification"] == marker["Classification"] == "UNKNOWN"
        assert row["source"] == marker["Source"] == "mpegts"
        assert row["ts"] == marker["Timestamp"]
        assert row["raw_json"] == json.dumps(marker, separators=(",", ":"), ensure_ascii=False)
    assert_redacted_diagnostics(result.stderr)


def test_root_alias_fixture_smoke_matches_monitor_command_shape() -> None:
    canonical = run_tidemark("monitor", FIXTURE, "--stream-type", "mpegts", "--json", "--timeout", "1")
    alias = run_tidemark(FIXTURE, "--stream-type", "mpegts", "--json", "--timeout", "1")

    assert canonical.returncode == 0, canonical.stderr
    assert alias.returncode == 0, alias.stderr
    canonical_markers = parse_ndjson(canonical.stdout)
    alias_markers = parse_ndjson(alias.stdout)
    assert len(alias_markers) == len(canonical_markers) > 0
    assert [comparable_marker(marker) for marker in alias_markers] == [
        comparable_marker(marker) for marker in canonical_markers
    ]
    assert_redacted_diagnostics(canonical.stderr)
    assert_redacted_diagnostics(alias.stderr)


def test_filter_id3_suppresses_scte35_fixture_stdout_json_out_and_db(tmp_path: Path) -> None:
    json_out = tmp_path / "markers.ndjson"
    db_path = tmp_path / "events.sqlite"

    result = run_tidemark(
        "monitor",
        FIXTURE,
        "--stream-type",
        "mpegts",
        "--json",
        "--filter",
        "id3",
        "--json-out",
        json_out,
        "--db",
        db_path,
        "--timeout",
        "1",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert json_out.read_text(encoding="utf-8") == ""
    assert read_ad_event_rows(db_path) == []
    assert "filtered=" in result.stderr
    assert_redacted_diagnostics(result.stderr)


def test_timeout_zero_stops_before_consuming_fixture_and_keeps_diagnostics_on_stderr_only(tmp_path: Path) -> None:
    json_out = tmp_path / "markers.ndjson"
    db_path = tmp_path / "events.sqlite"

    result = run_tidemark(
        "monitor",
        FIXTURE,
        "--stream-type",
        "mpegts",
        "--json",
        "--json-out",
        json_out,
        "--db",
        db_path,
        "--timeout",
        "0",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert json_out.read_text(encoding="utf-8") == ""
    assert read_ad_event_rows(db_path) == []
    assert result.stderr == "[tidemark] completed: reason=timeout markers=0 emitted=0 filtered=0\n"
    assert_redacted_diagnostics(result.stderr)
