from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tidemark.runtime.health import (
    DEFAULT_STALE_AFTER_SECONDS,
    SCHEMA_VERSION,
    HealthRecord,
    HealthReporter,
    RetryState,
    classify_record,
    create_reporter,
    format_status_report,
    make_run_id,
    read_status_entries,
    redact_source_label,
)


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_health_record_serializes_compact_schema_with_default_retry() -> None:
    started = utc("2026-04-28T12:00:00+00:00")
    heartbeat = started + timedelta(seconds=5)
    record = HealthRecord(
        run_id="monitor-abc123",
        command="monitor",
        source_label="example.com/stream",
        pid=1234,
        started_at=started,
        heartbeat_at=heartbeat,
        phase="polling",
        counters={"segments": 2},
        last_error="source unavailable: [redacted-url]",
    )

    payload = record.to_json_dict()

    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "run_id": "monitor-abc123",
        "command": "monitor",
        "source_label": "example.com/stream",
        "pid": 1234,
        "started_at": "2026-04-28T12:00:00Z",
        "heartbeat_at": "2026-04-28T12:00:05Z",
        "phase": "polling",
        "counters": {"segments": 2},
        "retry": {"attempt": 0, "next_retry_at": None, "last_retry_error": None},
        "last_error": "source unavailable: [redacted-url]",
        "terminal": False,
        "terminal_reason": None,
    }
    assert HealthRecord.from_json_dict(payload) == record
    assert RetryState().to_json_dict() == {"attempt": 0, "next_retry_at": None, "last_retry_error": None}


def test_make_run_id_is_filename_safe_and_reporter_uses_runs_directory(tmp_path: Path) -> None:
    run_id = make_run_id("monitor urls", now=utc("2026-04-28T12:00:00+00:00"), pid=77, entropy="abc/123")
    assert run_id == "monitor-urls-20260428T120000Z-77-abc-123"

    reporter = create_reporter(
        tmp_path,
        command="monitor urls",
        source="https://audio.example/live.m3u8?token=secret",
        run_id=run_id,
        pid=77,
        now=lambda: utc("2026-04-28T12:00:00+00:00"),
    )

    result = reporter.start(phase="starting")

    assert result.ok is True
    assert result.path == tmp_path / "runs" / f"{run_id}.json"
    assert result.path.exists()
    assert not (tmp_path / "status.json").exists()


def test_source_label_redaction_removes_queries_secret_values_and_private_dirs(tmp_path: Path) -> None:
    url_label = redact_source_label("https://user:pass@example.com/private/live.m3u8?token=secret&x=1")
    path_label = redact_source_label(str(tmp_path / "private" / "source.wav"))
    error_label = redact_source_label("failed api_key=abc123 url=https://example.test/live?token=secret")

    assert url_label == "example.com/private/live.m3u8"
    assert "token" not in url_label
    assert "secret" not in url_label
    assert path_label == "[local file]"
    assert str(tmp_path) not in path_label
    assert "abc123" not in error_label
    assert "token=secret" not in error_label
    assert "[redacted]" in error_label


def test_reporter_writes_atomically_updates_counters_and_reads_multiple_runs(tmp_path: Path) -> None:
    times = iter(
        [
            utc("2026-04-28T12:00:00+00:00"),
            utc("2026-04-28T12:00:05+00:00"),
            utc("2026-04-28T12:00:06+00:00"),
            utc("2026-04-28T12:00:07+00:00"),
        ]
    )
    reporter = create_reporter(
        tmp_path,
        command="monitor",
        source="https://example.test/a?token=secret",
        run_id="run-a",
        pid=100,
        now=lambda: next(times),
    )
    other = create_reporter(
        tmp_path,
        command="monitor",
        source="https://example.test/b",
        run_id="run-b",
        pid=101,
        now=lambda: utc("2026-04-28T12:00:08+00:00"),
    )

    assert reporter.start(counters={"segments": 1}).ok is True
    assert reporter.update(phase="polling", counters={"segments": 2, "errors": 1}).ok is True
    assert reporter.finish(reason="complete", counters={"segments": 3}).ok is True
    assert other.start(phase="polling").ok is True

    entries, diagnostics = read_status_entries(tmp_path, now=utc("2026-04-28T12:00:09+00:00"), pid_exists=lambda pid: True)

    assert diagnostics == []
    assert [entry.record.run_id for entry in entries] == ["run-a", "run-b"]
    assert entries[0].record.counters == {"segments": 3, "errors": 1}
    assert entries[0].record.terminal is True
    assert entries[0].record.terminal_reason == "complete"
    assert entries[0].status == "not-running"
    assert entries[1].status == "active"
    assert not list((tmp_path / "runs").glob("*.tmp"))


def test_read_status_entries_does_not_create_missing_directories(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "missing"

    entries, diagnostics = read_status_entries(runtime_dir)

    assert entries == []
    assert diagnostics == []
    assert not runtime_dir.exists()


def test_malformed_files_are_diagnostics_without_raw_payloads_or_private_paths(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "invalid.json").write_text('{"run_id": "abc", token=', encoding="utf-8")
    (runs / "missing-run.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8")
    (runs / "bad-counters.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": "bad-counters",
                "command": "monitor",
                "source_label": "safe",
                "pid": 1,
                "started_at": "2026-04-28T12:00:00Z",
                "heartbeat_at": "2026-04-28T12:00:00Z",
                "phase": "polling",
                "counters": ["bad"],
                "retry": {},
                "terminal": False,
            }
        ),
        encoding="utf-8",
    )
    (runs / "bad-time.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": "bad-time",
                "command": "monitor",
                "source_label": "https://example.test/live?token=secret",
                "pid": 1,
                "started_at": "not-a-time",
                "heartbeat_at": "2026-04-28T12:00:00Z",
                "phase": "polling",
                "counters": {},
                "retry": {},
                "terminal": False,
            }
        ),
        encoding="utf-8",
    )

    entries, diagnostics = read_status_entries(tmp_path, now=utc("2026-04-28T12:00:01+00:00"), pid_exists=lambda pid: True)

    assert entries == []
    assert [diagnostic.file_name for diagnostic in diagnostics] == [
        "bad-counters.json",
        "bad-time.json",
        "invalid.json",
        "missing-run.json",
    ]
    messages = "\n".join(d.message for d in diagnostics)
    assert "token=" not in messages
    assert "secret" not in messages
    assert str(tmp_path) not in messages
    assert "invalid JSON" in messages
    assert "missing required field" in messages
    assert "counters must be an object" in messages
    assert "timestamp is invalid" in messages


def test_classification_active_stale_not_running_dead_pid_terminal_and_bad_pid() -> None:
    now = utc("2026-04-28T12:10:00+00:00")
    fresh = HealthRecord(
        run_id="fresh",
        command="monitor",
        source_label="safe",
        pid=10,
        started_at=now - timedelta(seconds=10),
        heartbeat_at=now - timedelta(seconds=10),
        phase="polling",
    )
    old = HealthRecord(
        run_id="old",
        command="monitor",
        source_label="safe",
        pid=11,
        started_at=now - timedelta(seconds=DEFAULT_STALE_AFTER_SECONDS + 10),
        heartbeat_at=now - timedelta(seconds=DEFAULT_STALE_AFTER_SECONDS + 10),
        phase="polling",
    )
    dead = HealthRecord(
        run_id="dead",
        command="monitor",
        source_label="safe",
        pid=12,
        started_at=now,
        heartbeat_at=now,
        phase="polling",
    )
    terminal = HealthRecord(
        run_id="terminal",
        command="monitor",
        source_label="safe",
        pid=13,
        started_at=now,
        heartbeat_at=now,
        phase="finished",
        terminal=True,
        terminal_reason="complete",
    )
    bad_pid = HealthRecord(
        run_id="bad-pid",
        command="monitor",
        source_label="safe",
        pid=0,
        started_at=now,
        heartbeat_at=now,
        phase="polling",
    )

    live_pids = {10, 11, 13}

    assert classify_record(fresh, now=now, pid_exists=lambda pid: pid in live_pids).status == "active"
    assert classify_record(old, now=now, pid_exists=lambda pid: pid in live_pids).status == "stale"
    assert classify_record(dead, now=now, pid_exists=lambda pid: pid in live_pids).status == "not-running"
    assert classify_record(terminal, now=now, pid_exists=lambda pid: pid in live_pids).status == "not-running"
    bad_pid_entry = classify_record(bad_pid, now=now, pid_exists=lambda pid: True)
    assert bad_pid_entry.status == "not-running"
    assert any("invalid pid" in diagnostic.message for diagnostic in bad_pid_entry.diagnostics)


def test_pid_probe_errors_classify_conservatively_as_stale_or_not_running() -> None:
    now = utc("2026-04-28T12:10:00+00:00")
    fresh = HealthRecord(
        run_id="fresh",
        command="monitor",
        source_label="safe",
        pid=10,
        started_at=now,
        heartbeat_at=now,
        phase="polling",
    )
    old = HealthRecord(
        run_id="old",
        command="monitor",
        source_label="safe",
        pid=11,
        started_at=now - timedelta(seconds=DEFAULT_STALE_AFTER_SECONDS + 1),
        heartbeat_at=now - timedelta(seconds=DEFAULT_STALE_AFTER_SECONDS + 1),
        phase="polling",
    )

    def broken_probe(pid: int) -> bool:
        raise OSError(f"private failure for {pid}")

    fresh_entry = classify_record(fresh, now=now, pid_exists=broken_probe)
    old_entry = classify_record(old, now=now, pid_exists=broken_probe)

    assert fresh_entry.status == "not-running"
    assert old_entry.status == "stale"
    diagnostics = "\n".join(d.message for d in fresh_entry.diagnostics + old_entry.diagnostics)
    assert "private failure" not in diagnostics
    assert "pid liveness probe failed" in diagnostics


def test_reporter_retry_persists_redacted_retry_state_and_preserves_counters(tmp_path: Path) -> None:
    reporter = create_reporter(
        tmp_path,
        command="monitor",
        source="https://example.test/live?token=secret",
        run_id="retry-run",
        pid=100,
        now=lambda: utc("2026-04-28T12:00:00+00:00"),
    )
    assert reporter.start(phase="polling", counters={"segments": 2}).ok is True

    result = reporter.retry(
        attempt=2,
        next_retry_at=utc("2026-04-28T12:00:05+00:00"),
        error="transient api_key=secret from https://example.test/live?token=secret",
        phase="retrying",
        counters={"errors": 1},
    )

    assert result.ok is True
    entries, diagnostics = read_status_entries(
        tmp_path,
        now=utc("2026-04-28T12:00:01+00:00"),
        pid_exists=lambda pid: True,
    )
    assert diagnostics == []
    assert len(entries) == 1
    record = entries[0].record
    assert record.phase == "retrying"
    assert record.counters == {"segments": 2, "errors": 1}
    assert record.retry == RetryState(
        attempt=2,
        next_retry_at=utc("2026-04-28T12:00:05+00:00"),
        last_retry_error="transient api_key=[redacted] from example.test/live",
    )
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["retry"] == {
        "attempt": 2,
        "next_retry_at": "2026-04-28T12:00:05Z",
        "last_retry_error": "transient api_key=[redacted] from example.test/live",
    }


def test_format_status_report_retry_output_remains_compact_and_redacted(tmp_path: Path) -> None:
    record = HealthRecord(
        run_id="retry-run",
        command="monitor",
        source_label="source",
        pid=100,
        started_at=utc("2026-04-28T12:00:00+00:00"),
        heartbeat_at=utc("2026-04-28T12:00:01+00:00"),
        phase="retrying",
        counters={"errors": 1},
        retry=RetryState(
            attempt=3,
            next_retry_at=utc("2026-04-28T12:00:10+00:00"),
            last_retry_error="raw token=secret from https://example.test/live?token=secret",
        ),
    )
    entry = classify_record(record, now=utc("2026-04-28T12:00:02+00:00"), pid_exists=lambda pid: True)

    report = format_status_report([entry], [], runtime_dir=tmp_path)

    assert "retry=attempt=3,next=2026-04-28T12:00:10Z,last_error=raw token=[redacted] from example.test/live" in report
    assert "token=secret" not in report
    assert str(tmp_path) not in report


def test_reporter_retry_write_failure_is_returned_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reporter = create_reporter(
        tmp_path,
        command="monitor",
        source="safe",
        run_id="retry-run",
        pid=100,
        now=lambda: utc("2026-04-28T12:00:00+00:00"),
    )

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("/private/path raw-token=secret")

    monkeypatch.setattr(Path, "replace", fail_replace)

    result = reporter.retry(
        attempt=1,
        next_retry_at=utc("2026-04-28T12:00:05+00:00"),
        error="raw token=secret",
    )

    assert result.ok is False
    assert result.diagnostic is not None
    assert result.diagnostic.file_name == "retry-run.json"
    assert "secret" not in result.diagnostic.message
    assert reporter.record.retry.attempt == 1


def test_retry_state_rejects_non_datetime_next_retry_values() -> None:
    with pytest.raises(ValueError):
        RetryState(attempt=1, next_retry_at="not-a-datetime")  # type: ignore[arg-type]


def test_reporter_write_failure_is_returned_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reporter = HealthReporter(
        runtime_dir=tmp_path,
        record=HealthRecord(
            run_id="run-a",
            command="monitor",
            source_label="safe",
            pid=1,
            started_at=utc("2026-04-28T12:00:00+00:00"),
            heartbeat_at=utc("2026-04-28T12:00:00+00:00"),
            phase="starting",
        ),
        now=lambda: utc("2026-04-28T12:00:01+00:00"),
    )

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("/private/path raw-token=secret")

    monkeypatch.setattr(Path, "replace", fail_replace)

    result = reporter.update(last_error="raw token=secret from https://example.test/live?token=secret")

    assert result.ok is False
    assert result.diagnostic is not None
    assert result.diagnostic.file_name == "run-a.json"
    assert "secret" not in result.diagnostic.message
    assert "token=secret" not in result.diagnostic.message
    assert str(tmp_path) not in result.diagnostic.message
    assert reporter.record.last_error is not None
    assert "token=secret" not in reporter.record.last_error
