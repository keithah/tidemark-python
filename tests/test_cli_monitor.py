from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.markers import UNKNOWN, AdMarker
from tidemark.monitor import MonitorOptions, MonitorResult
from tidemark.monitor_sources import MonitorSourceError, StreamType


runner = CliRunner()


def invoke(args: list[str]):
    from tidemark.cli.main import app

    return runner.invoke(app, args)


def marker(
    marker_type: str = "HLS",
    *,
    tag: str | None = "#EXT-X-CUE-OUT",
    source: str = "fixture?token=secret",
    classification: str = UNKNOWN,
    timestamp: float = 1.0,
    fields: dict[str, object] | None = None,
) -> AdMarker:
    return AdMarker(
        type=marker_type,
        classification=classification,
        source=source,
        tag=tag,
        timestamp=timestamp,
        fields=fields or {},
        raw_base64="secret-payload",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def only_run_file(runtime_dir: Path) -> dict[str, object]:
    run_files = sorted((runtime_dir / "runs").glob("*.json"))
    assert len(run_files) == 1
    return json.loads(run_files[0].read_text(encoding="utf-8"))


def patch_success(monkeypatch: pytest.MonkeyPatch, *, stdout_line: str | None = None):
    calls: list[dict[str, object]] = []

    def fake_monitor_source(source, *, stream_type="auto", timeout=None, **kwargs):
        calls.append(
            {
                "source": source,
                "stream_type": stream_type,
                "timeout": timeout,
                "headers": kwargs.get("headers"),
                "follow_live_hls": kwargs.get("follow_live_hls"),
            }
        )
        return iter(())

    def fake_run_monitor(marker_source, *, options: MonitorOptions, stdout, stderr):
        source_iter = marker_source() if callable(marker_source) else marker_source
        calls.append(
            {
                "marker_source": source_iter,
                "options": options,
            }
        )
        if stdout_line is not None:
            stdout.write(stdout_line)
            stdout.write("\n")
        if options.emit_summary:
            stderr.write("[tidemark] completed: reason=eof markers=0 emitted=0 filtered=0\n")
        return MonitorResult(reason="eof", markers_seen=0, markers_emitted=0, markers_filtered=0)

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", fake_monitor_source)
    monkeypatch.setattr("tidemark.cli.cmd_monitor.run_monitor", fake_run_monitor)
    return calls


def test_monitor_command_invokes_library_once_with_default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(["monitor", "tests/fixtures/scte35_splice_null.ts"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[0] == {
        "source": "tests/fixtures/scte35_splice_null.ts",
        "stream_type": "auto",
        "timeout": None,
        "headers": None,
        "follow_live_hls": True,
    }
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.source_url == "tests/fixtures/scte35_splice_null.ts"
    assert options.marker_filter is None
    assert options.json_out is None
    assert options.db_path is None
    assert options.timeout is None
    assert options.emit_summary is True
    assert options.retry_policy is not None
    assert options.retry_policy.max_attempts == 0


def test_monitor_command_retries_transient_setup_and_records_lifecycle_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0
    emitted = marker()
    runtime_dir = tmp_path / "runtime"
    log_file = tmp_path / "lifecycle.jsonl"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'runtime_dir = "{runtime_dir}"',
                f'log_file = "{log_file}"',
                "[monitor]",
                "retry_attempts = 1",
                "retry_initial_backoff_seconds = 0.0",
                "retry_max_backoff_seconds = 1.0",
            ]
        ),
        encoding="utf-8",
    )

    def flaky_source(source, *, stream_type="auto", timeout=None, **kwargs):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MonitorSourceError(
                "hls source setup failed https://example.test/live.m3u8?token=secret local=/home/keith/private.ts",
                stream_type=StreamType.HLS,
                phase="setup",
            )
        return iter([emitted])

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", flaky_source)

    result = invoke(["monitor", "https://example.test/live.m3u8?token=secret", "--config", str(config_path), "--quiet"])

    assert result.exit_code == 0, result.output
    assert attempts == 2
    assert result.stdout.splitlines() == [emitted.to_json()]
    assert result.stderr == ""
    status = only_run_file(runtime_dir)
    assert status["phase"] == "completed"
    assert status["terminal"] is True
    assert status["terminal_reason"] == "eof"
    assert status["retry"]["attempt"] == 1
    assert status["retry"]["last_retry_error"] == "hls source setup failed example.test/live.m3u8 local=private.ts"
    assert "token=secret" not in json.dumps(status)
    events = read_jsonl(log_file)
    assert [event["event"] for event in events] == ["monitor.start", "monitor.retry", "monitor.reconnect", "monitor.terminal"]
    assert events[1]["retry_attempt"] == 1
    assert events[1]["error"] == "hls source setup failed example.test/live.m3u8 local=private.ts"
    assert events[-1]["terminal_reason"] == "eof"
    assert "token=secret" not in log_file.read_text(encoding="utf-8")
    assert "/home/keith" not in log_file.read_text(encoding="utf-8")


def test_monitor_command_exhausted_retry_records_redacted_status_log_and_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0
    runtime_dir = tmp_path / "runtime"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'runtime_dir = "{runtime_dir}"',
                "[monitor]",
                "retry_attempts = 1",
                "retry_initial_backoff_seconds = 0.0",
                "retry_max_backoff_seconds = 1.0",
            ]
        ),
        encoding="utf-8",
    )

    def failing_source(source, *, stream_type="auto", timeout=None, **kwargs):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        raise MonitorSourceError(
            "icy source setup failed https://example.test/live?token=secret file=/Users/alice/private/live.ts",
            stream_type=StreamType.ICY,
            phase="setup",
        )

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", failing_source)

    result = invoke(["monitor", "https://example.test/live?token=secret", "--config", str(config_path)])

    assert result.exit_code == 1
    assert attempts == 2
    assert result.stdout == ""
    assert "[tidemark] error:" in result.stderr
    assert "token=secret" not in result.stderr
    assert "/Users/alice" not in result.stderr
    status_result = invoke(["status", "--runtime-dir", str(runtime_dir)])
    assert status_result.exit_code == 0, status_result.output
    assert "retry=attempt=1,next=" in status_result.stdout
    assert "last_error=icy source setup failed example.test/live file=live.ts" in status_result.stdout
    assert "terminal=error" in status_result.stdout
    assert "token=secret" not in status_result.stdout
    assert "/Users/alice" not in status_result.stdout
    log_file = runtime_dir / "logs" / "tidemark.jsonl"
    log_text = log_file.read_text(encoding="utf-8")
    assert "monitor.retry" in log_text
    assert "monitor.terminal" in log_text
    assert "token=secret" not in log_text
    assert "/Users/alice" not in log_text


def test_monitor_command_retry_disabled_by_default_and_env_log_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0
    runtime_dir = tmp_path / "runtime"
    env_log_file = tmp_path / "env-log.jsonl"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(f'[paths]\nruntime_dir = "{runtime_dir}"\n', encoding="utf-8")

    def failing_source(source, *, stream_type="auto", timeout=None, **kwargs):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        raise MonitorSourceError("hls source setup failed token=secret", stream_type=StreamType.HLS, phase="setup")

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", failing_source)

    result = runner.invoke(
        __import__("tidemark.cli.main", fromlist=["app"]).app,
        ["monitor", "https://example.test/live?token=secret", "--config", str(config_path), "--quiet"],
        env={"TIDEMARK_LOG_FILE": str(env_log_file)},
    )

    assert result.exit_code == 1
    assert attempts == 1
    assert result.stdout == ""
    assert result.stderr == "[tidemark] error: hls source setup failed token=[redacted]\n"
    assert env_log_file.exists()
    assert not (runtime_dir / "logs" / "tidemark.jsonl").exists()


class RecordingReporter:
    def __init__(self, events: list[tuple[str, dict[str, object]]]) -> None:
        self.events = events

    def start(self, **kwargs):
        self.events.append(("start", kwargs))

    def update(self, **kwargs):
        self.events.append(("update", kwargs))

    def finish(self, **kwargs):
        self.events.append(("finish", kwargs))

    def fail(self, error, **kwargs):
        self.events.append(("fail", {"error": error, **kwargs}))


def test_monitor_command_creates_reporter_before_source_and_passes_progress_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    calls: list[str] = []
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(f'[paths]\nruntime_dir = "{tmp_path / "runtime"}"\n', encoding="utf-8")

    def fake_create_reporter(runtime_dir, *, command, source, **kwargs):
        calls.append("create_reporter")
        assert runtime_dir == tmp_path / "runtime"
        assert command == "monitor"
        assert source == "http://example.test/live.m3u8?token=secret"
        return RecordingReporter(events)

    def fake_monitor_source(source, **kwargs):  # noqa: ARG001
        calls.append("monitor_source")
        return iter(())

    def fake_run_monitor(marker_source, *, options: MonitorOptions, stdout, stderr):  # noqa: ARG001
        calls.append("run_monitor")
        assert options.progress_callback is not None
        options.progress_callback(
            type(
                "Progress",
                (),
                {
                    "phase": "running",
                    "reason": None,
                    "counters": {"markers_seen": 1, "markers_emitted": 1, "markers_filtered": 0, "sink_warnings": 0},
                    "error": None,
                },
            )()
        )
        return MonitorResult(reason="eof", markers_seen=1, markers_emitted=1, markers_filtered=0)

    monkeypatch.setattr("tidemark.cli.cmd_monitor.create_reporter", fake_create_reporter)
    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", fake_monitor_source)
    monkeypatch.setattr("tidemark.cli.cmd_monitor.run_monitor", fake_run_monitor)

    result = invoke(["monitor", "http://example.test/live.m3u8?token=secret", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert result.stderr == ""
    assert calls == ["create_reporter", "run_monitor"]
    assert events == [
        ("start", {"phase": "setup", "counters": {"markers_seen": 0, "markers_emitted": 0, "markers_filtered": 0, "sink_warnings": 0}}),
        ("update", {"phase": "running", "counters": {"markers_seen": 1, "markers_emitted": 1, "markers_filtered": 0, "sink_warnings": 0}}),
        ("finish", {"phase": "completed", "reason": "eof", "counters": {"markers_seen": 1, "markers_emitted": 1, "markers_filtered": 0, "sink_warnings": 0}}),
    ]


def test_monitor_command_records_source_setup_failure_without_leaking_or_changing_fatal_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tidemark.monitor_sources import MonitorSourceError

    events: list[tuple[str, dict[str, object]]] = []
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(f'[paths]\nruntime_dir = "{tmp_path / "runtime"}"\n', encoding="utf-8")

    monkeypatch.setattr("tidemark.cli.cmd_monitor.create_reporter", lambda *args, **kwargs: RecordingReporter(events))

    def fail_source(source, **kwargs):  # noqa: ARG001
        raise MonitorSourceError("source setup failed: http://example.test/live.m3u8?token=secret")

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", fail_source)

    result = invoke(["monitor", "http://example.test/live.m3u8?token=secret", "--config", str(config_path), "--quiet"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error:" in result.stderr
    assert "token=secret" not in result.stderr
    assert events == [
        ("start", {"phase": "setup", "counters": {"markers_seen": 0, "markers_emitted": 0, "markers_filtered": 0, "sink_warnings": 0}}),
        ("update", {"phase": "running", "counters": {"markers_seen": 0, "markers_emitted": 0, "markers_filtered": 0, "sink_warnings": 0}}),
        (
            "fail",
            {
                "error": "source setup failed: example.test/live.m3u8",
                "phase": "error",
                "reason": "error",
                "counters": {"markers_seen": 0, "markers_emitted": 0, "markers_filtered": 0, "sink_warnings": 0},
            },
        ),
    ]


def test_root_url_alias_invokes_same_monitor_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(["udp://239.1.1.1:5000", "--stream-type", "udp", "--timeout", "2.5"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[0]["source"] == "udp://239.1.1.1:5000"
    assert calls[0]["stream_type"] == "udp"
    assert calls[0]["timeout"] == 2.5
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.source_url == "udp://239.1.1.1:5000"
    assert options.timeout == 2.5


def test_root_url_alias_accepts_config_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_success(monkeypatch)
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text('[monitor]\nstream_type = "udp"\ntimeout_seconds = 2.5\n', encoding="utf-8")

    result = invoke(["udp://239.1.1.1:5000", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert calls[0]["stream_type"] == "udp"
    assert calls[0]["timeout"] == 2.5
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.timeout == 2.5


def test_monitor_command_passes_all_options_without_suppressing_ndjson(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_success(monkeypatch, stdout_line='{"Type":"SCTE35"}')
    json_out = tmp_path / "markers.ndjson"
    db_path = tmp_path / "events.sqlite"

    result = invoke(
        [
            "monitor",
            "http://example.test/live.m3u8?token=secret",
            "--stream-type",
            "hls",
            "--json",
            "--quiet",
            "--filter",
            "scte35",
            "--json-out",
            str(json_out),
            "--timeout",
            "3.25",
            "--db",
            str(db_path),
        ]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == '{"Type":"SCTE35"}\n'
    assert result.stderr == ""
    assert calls[0]["stream_type"] == "hls"
    assert calls[0]["timeout"] == 3.25
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.source_url == "http://example.test/live.m3u8?token=secret"
    assert options.marker_filter == "scte35"
    assert options.json_out == json_out
    assert options.db_path == db_path
    assert options.timeout == 3.25
    assert options.emit_summary is False


def test_root_alias_does_not_run_when_monitor_subcommand_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(["monitor", "fixture.ts", "--stream-type", "mpegts"])

    assert result.exit_code == 0, result.output
    assert [call.get("source") for call in calls if "source" in call] == ["fixture.ts"]


def test_status_is_real_command_not_monitor_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(["status", "--runtime-dir", str(tmp_path / "missing-runtime")])

    assert result.exit_code == 0, result.output
    assert "No runtime health records found" in result.stdout
    assert calls == []


@pytest.mark.parametrize("args", [["--help"], ["monitor", "--help"]])
def test_help_paths_do_not_invoke_monitor(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(args)

    assert result.exit_code == 0
    assert "monitor" in result.stdout.lower()
    assert calls == []


def test_no_url_is_usage_error_before_monitor_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(["monitor"])

    assert result.exit_code != 0
    assert calls == []


@pytest.mark.parametrize(
    "args",
    [
        ["monitor", "fixture.ts", "--stream-type", "dash"],
        ["monitor", "fixture.ts", "--filter", "ad"],
        ["monitor", "fixture.ts", "--timeout", "-1"],
    ],
)
def test_invalid_cli_values_are_rejected_before_monitor_starts(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    calls = patch_success(monkeypatch)

    result = invoke(args)

    assert result.exit_code != 0
    assert calls == []


def test_monitor_error_result_exits_one_and_keeps_redacted_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_monitor_source(source, **kwargs):
        return iter(())

    def fake_run_monitor(marker_source, *, options: MonitorOptions, stdout, stderr):
        stderr.write("[tidemark] error: source setup failed\n")
        return MonitorResult(reason="error", markers_seen=0, markers_emitted=0, markers_filtered=0, error="source setup failed")

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", fake_monitor_source)
    monkeypatch.setattr("tidemark.cli.cmd_monitor.run_monitor", fake_run_monitor)

    result = invoke(["monitor", "http://example.test/live.ts?token=secret"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: source setup failed" in result.stderr
    assert "token=secret" not in result.stderr
    assert "Traceback" not in result.stderr
