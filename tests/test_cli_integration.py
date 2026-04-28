from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path


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
