from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from tidemark.runtime.health import HealthRecord, RetryState


runner = CliRunner()


def invoke(args: list[str], *, env: dict[str, str] | None = None):
    from tidemark.cli.main import app

    return runner.invoke(app, args, env=env)


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def write_record(runtime_dir: Path, record: HealthRecord) -> Path:
    runs_dir = runtime_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.run_id}.json"
    path.write_text(json.dumps(record.to_json_dict()), encoding="utf-8")
    return path


def record(
    run_id: str,
    *,
    command: str = "monitor",
    pid: int = 1234,
    heartbeat_at: datetime | None = None,
    phase: str = "polling",
    counters: dict[str, int | float] | None = None,
    retry: RetryState | None = None,
    last_error: str | None = None,
    terminal: bool = False,
    terminal_reason: str | None = None,
    source_label: str = "https://example.test/live.m3u8?token=secret",
) -> HealthRecord:
    started = utc("2026-04-28T12:00:00+00:00")
    return HealthRecord(
        run_id=run_id,
        command=command,
        source_label=source_label,
        pid=pid,
        started_at=started,
        heartbeat_at=heartbeat_at or datetime.now(timezone.utc),
        phase=phase,
        counters=counters or {},
        retry=retry or RetryState(),
        last_error=last_error,
        terminal=terminal,
        terminal_reason=terminal_reason,
    )


def test_status_empty_runtime_dir_exits_zero_without_creating_directories(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "missing-runtime"

    result = invoke(["status", "--runtime-dir", str(runtime_dir)])

    assert result.exit_code == 0, result.output
    assert "No runtime health records found" in result.stdout
    assert "not running" in result.stdout.lower()
    assert not runtime_dir.exists()


def test_status_uses_config_env_and_cli_runtime_dir_precedence(tmp_path: Path) -> None:
    config_runtime = tmp_path / "config-runtime"
    env_runtime = tmp_path / "env-runtime"
    cli_runtime = tmp_path / "cli-runtime"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(f'[paths]\nruntime_dir = "{config_runtime}"\n', encoding="utf-8")
    write_record(config_runtime, record("from-config", command="monitor"))
    write_record(env_runtime, record("from-env", command="ingest"))
    write_record(cli_runtime, record("from-cli", command="monitor"))

    config_result = invoke(["status", "--config", str(config_path)])
    env_result = invoke(
        ["status", "--config", str(config_path)],
        env={"TIDEMARK_RUNTIME_DIR": str(env_runtime)},
    )
    cli_result = invoke(
        ["status", "--config", str(config_path), "--runtime-dir", str(cli_runtime)],
        env={"TIDEMARK_RUNTIME_DIR": str(env_runtime)},
    )

    assert config_result.exit_code == 0, config_result.output
    assert "from-config" in config_result.stdout
    assert "from-env" not in config_result.stdout
    assert env_result.exit_code == 0, env_result.output
    assert "from-env" in env_result.stdout
    assert "from-config" not in env_result.stdout
    assert cli_result.exit_code == 0, cli_result.output
    assert "from-cli" in cli_result.stdout
    assert "from-env" not in cli_result.stdout


def test_status_lists_multiple_records_with_state_details_and_redaction(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "runtime"
    stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=999)
    write_record(
        runtime_dir,
        record(
            "active-run",
            command="monitor",
            pid=100,
            phase="polling",
            counters={"segments": 3, "errors": 1},
            retry=RetryState(attempt=2, last_retry_error="token=secret retry"),
        ),
    )
    write_record(
        runtime_dir,
        record("stale-run", command="ingest", pid=101, heartbeat_at=stale_heartbeat, phase="transcribing"),
    )
    write_record(runtime_dir, record("dead-run", command="monitor", pid=102, phase="polling"))
    write_record(
        runtime_dir,
        record(
            "terminal-run",
            command="ingest",
            pid=103,
            phase="failed",
            last_error="failed url=https://example.test/live?token=secret api_key=abc123",
            terminal=True,
            terminal_reason="error",
        ),
    )

    monkeypatch.setattr("tidemark.runtime.health.pid_exists", lambda pid: pid in {100, 101, 103})

    result = invoke(["status", "--runtime-dir", str(runtime_dir)])

    assert result.exit_code == 0, result.output
    output = result.stdout
    assert "active-run" in output and "state=active" in output
    assert "stale-run" in output and "state=stale" in output
    assert "dead-run" in output and "state=not-running" in output
    assert "terminal-run" in output and "terminal=error" in output
    assert "command=monitor" in output
    assert "command=ingest" in output
    assert "pid=100" in output
    assert "phase=polling" in output
    assert "counters=errors=1,segments=3" in output
    assert "retry=attempt=2" in output
    assert "last_error=" in output
    assert "token=secret" not in output
    assert "abc123" not in output


def test_status_prints_ingest_restart_counters_with_sorted_numeric_formatter(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    write_record(
        runtime_dir,
        record(
            "ingest-restart-run",
            command="ingest",
            phase="completed",
            counters={
                "segments": 5,
                "processed": 2,
                "skipped": 3,
                "failed": 1,
                "words": 42,
                "markers": 4,
                "issues": 1,
                "retained": 2,
                "songs": 1,
            },
            terminal=True,
            terminal_reason="finished",
            source_label="fixture://private.example/live.m3u8?token=secret",
        ),
    )

    result = invoke(["status", "--runtime-dir", str(runtime_dir)])

    assert result.exit_code == 0, result.output
    assert "ingest-restart-run" in result.stdout
    assert "command=ingest" in result.stdout
    assert (
        "counters=failed=1,issues=1,markers=4,processed=2,retained=2,segments=5,skipped=3,songs=1,words=42"
        in result.stdout
    )
    assert "token=secret" not in result.stdout
    assert str(runtime_dir) not in result.stdout



def test_status_reports_malformed_files_as_diagnostics_without_crashing_or_leaking(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    runs_dir = runtime_dir / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "bad.json").write_text('{"run_id": "bad", token=secret', encoding="utf-8")
    (runs_dir / "wrong-schema.json").write_text(json.dumps({"schema_version": 999, "token": "secret"}), encoding="utf-8")
    write_record(runtime_dir, record("good-run", source_label="https://example.test/live?token=secret"))

    result = invoke(["status", "--runtime-dir", str(runtime_dir)])

    assert result.exit_code == 0, result.output
    assert "good-run" in result.stdout
    assert "Diagnostics: 2 malformed status file(s) skipped" in result.stdout
    assert "bad.json: invalid JSON in status file" in result.stdout
    assert "wrong-schema.json: schema version is unsupported" in result.stdout
    assert "token=secret" not in result.stdout
    assert "secret" not in result.stderr


def test_status_invalid_config_path_uses_redacted_tidemark_error(tmp_path: Path) -> None:
    missing_config = tmp_path / "private" / "tidemark.toml"

    result = invoke(["status", "--config", str(missing_config)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error:" in result.stderr
    assert "tidemark.toml" in result.stderr
    assert str(tmp_path) not in result.stderr
    assert "Traceback" not in result.stderr


def test_status_does_not_create_absent_configured_db_path(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    db_path = tmp_path / "absent.sqlite"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(
        f'[paths]\nruntime_dir = "{runtime_dir}"\ndb = "{db_path}"\n',
        encoding="utf-8",
    )

    result = invoke(["status", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "No runtime health records found" in result.stdout
    assert not runtime_dir.exists()
    assert not db_path.exists()
