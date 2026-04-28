from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tidemark.config import PathsConfig, TidemarkConfig
from tidemark.runtime.logging import (
    LifecycleLogger,
    resolve_lifecycle_log_path,
    write_lifecycle_event,
)


FIXED_NOW = datetime(2026, 4, 28, 20, 0, 1, tzinfo=timezone.utc)


def read_json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_write_lifecycle_event_appends_compact_sorted_jsonl_with_optional_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "tidemark.jsonl"
    next_retry_at = datetime(2026, 4, 28, 20, 0, 2, tzinfo=timezone.utc)

    first = write_lifecycle_event(
        log_path,
        event="monitor-start",
        command="monitor",
        run_id="run-1",
        source_label="https://example.test/stream?token=secret",
        phase="starting",
        counters={"markers": 2, "latency": 1.5, "ignored": "bad", "flag": True},
        retry_attempt=1,
        next_retry_at=next_retry_at,
        latency_ms=12.25,
        error="Bearer top-secret-token",
        terminal_reason="finished token=terminal-secret",
        now=lambda: FIXED_NOW,
    )
    second = write_lifecycle_event(
        log_path,
        event="monitor-retry",
        command="monitor",
        run_id="run-1",
        source_label="source",
        phase="retrying",
        now=lambda: FIXED_NOW,
    )

    assert first.ok is True
    assert second.ok is True
    raw_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    assert raw_lines[0] == (
        '{"command":"monitor","counters":{"latency":1.5,"markers":2},"error":"Bearer [redacted]",'
        '"event":"monitor-start","latency_ms":12.25,"next_retry_at":"2026-04-28T20:00:02Z",'
        '"phase":"starting","retry_attempt":1,"run_id":"run-1","source_label":"example.test/stream",'
        '"terminal_reason":"finished token=[redacted]","ts":"2026-04-28T20:00:01Z"}'
    )
    assert read_json_lines(log_path)[1] == {
        "command": "monitor",
        "event": "monitor-retry",
        "phase": "retrying",
        "run_id": "run-1",
        "source_label": "source",
        "ts": "2026-04-28T20:00:01Z",
    }


def test_lifecycle_logger_redacts_secret_bearing_inputs(tmp_path: Path) -> None:
    log_path = tmp_path / "tidemark.jsonl"
    raw_parent = str(tmp_path / "private" / "source.mp3")
    logger = LifecycleLogger(log_path, now=lambda: FIXED_NOW)

    result = logger.write(
        event="monitor-failed",
        command="monitor",
        run_id="run-2",
        source_label=f"https://user:password@example.test/private.m3u8?api_key=query-secret&token=query-token",
        phase="failed",
        error=f"failed at {raw_parent} with token=inline-secret and Bearer header-secret",
        terminal_reason=f"terminal Basic basic-secret {raw_parent}",
    )

    assert result.ok is True
    contents = log_path.read_text(encoding="utf-8")
    for raw in [
        "password",
        "query-secret",
        "query-token",
        "inline-secret",
        "header-secret",
        "basic-secret",
        str(tmp_path),
        "private/source.mp3",
    ]:
        assert raw not in contents
    event = read_json_lines(log_path)[0]
    assert event["source_label"] == "example.test/private.m3u8"
    assert event["error"] == "failed at source.mp3 with token=[redacted] and Bearer [redacted]"
    assert event["terminal_reason"] == "terminal Basic [redacted] source.mp3"


def test_lifecycle_logging_is_best_effort_for_invalid_parent_path(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("already here", encoding="utf-8")
    log_path = parent_file / "tidemark.jsonl"

    result = write_lifecycle_event(
        log_path,
        event="monitor-start",
        command="monitor",
        run_id="run-3",
        source_label="source",
        phase="starting",
        now=lambda: FIXED_NOW,
    )

    assert result.ok is False
    assert not log_path.exists()
    assert parent_file.read_text(encoding="utf-8") == "already here"


def test_lifecycle_logging_is_best_effort_for_unserializable_inputs(tmp_path: Path) -> None:
    log_path = tmp_path / "tidemark.jsonl"

    result = write_lifecycle_event(
        log_path,
        event="monitor-failed",
        command="monitor",
        run_id="run-4",
        source_label=object(),
        phase="failed",
        error=RuntimeError("Bearer unusual-secret"),
        counters={"valid": 1, "bad": object()},
        terminal_reason=object(),
        now=lambda: FIXED_NOW,
    )

    assert result.ok is True
    payload = read_json_lines(log_path)[0]
    assert payload["counters"] == {"valid": 1}
    assert "unusual-secret" not in log_path.read_text(encoding="utf-8")
    assert isinstance(payload["source_label"], str)
    assert isinstance(payload["terminal_reason"], str)


def test_resolve_lifecycle_log_path_prefers_explicit_log_file(tmp_path: Path) -> None:
    config = TidemarkConfig(paths=PathsConfig(runtime_dir=tmp_path / "runtime", log_file=tmp_path / "custom.jsonl"))

    assert resolve_lifecycle_log_path(config) == tmp_path / "custom.jsonl"


def test_resolve_lifecycle_log_path_defaults_under_runtime_dir(tmp_path: Path) -> None:
    config = TidemarkConfig(paths=PathsConfig(runtime_dir=tmp_path / "runtime", log_file=None))

    assert resolve_lifecycle_log_path(config) == tmp_path / "runtime" / "logs" / "tidemark.jsonl"
