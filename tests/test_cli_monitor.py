from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.monitor import MonitorOptions, MonitorResult


runner = CliRunner()


def invoke(args: list[str]):
    from tidemark.cli.main import app

    return runner.invoke(app, args)


def patch_success(monkeypatch: pytest.MonkeyPatch, *, stdout_line: str | None = None):
    calls: list[dict[str, object]] = []

    def fake_monitor_source(source, *, stream_type="auto", timeout=None, **kwargs):
        calls.append(
            {
                "source": source,
                "stream_type": stream_type,
                "timeout": timeout,
                "headers": kwargs.get("headers"),
            }
        )
        return iter(())

    def fake_run_monitor(marker_source, *, options: MonitorOptions, stdout, stderr):
        calls.append(
            {
                "marker_source": marker_source,
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
    }
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.source_url == "tests/fixtures/scte35_splice_null.ts"
    assert options.marker_filter is None
    assert options.json_out is None
    assert options.db_path is None
    assert options.timeout is None
    assert options.emit_summary is True


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
