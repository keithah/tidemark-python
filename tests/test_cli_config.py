from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.ingest.pipeline import IngestPipelineResult
from tidemark.monitor import MonitorOptions, MonitorResult


runner = CliRunner()


def invoke(args: list[str], *, env: dict[str, str] | None = None):
    from tidemark.cli.main import app

    return runner.invoke(app, args, env=env)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "private-tidemark.toml"
    path.write_text(body.strip(), encoding="utf-8")
    return path


def patch_monitor(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_monitor_source(source, *, stream_type="auto", timeout=None, **kwargs):
        calls.append({"source": source, "stream_type": stream_type, "timeout": timeout})
        return iter(())

    def fake_run_monitor(marker_source, *, options: MonitorOptions, stdout, stderr):
        calls.append({"marker_source": marker_source, "options": options})
        return MonitorResult(reason="eof", markers_seen=0, markers_emitted=0, markers_filtered=0)

    monkeypatch.setattr("tidemark.cli.cmd_monitor.monitor_source", fake_monitor_source)
    monkeypatch.setattr("tidemark.cli.cmd_monitor.run_monitor", fake_run_monitor)
    return calls


def patch_ingest(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_ingest_source_to_db(source, **kwargs):
        calls.append({"source": Path(source), **kwargs})
        return IngestPipelineResult(segment_ids=(1,), transcript_word_ids=(), ad_event_ids=(), issues=())

    monkeypatch.setattr("tidemark.cli.cmd_ingest.ingest_source_to_db", fake_ingest_source_to_db)
    return calls


def write_manifest(path: Path) -> Path:
    (path.parent / "segment.ts").write_bytes(b"not decoded")
    path.write_text("#EXTM3U\n#EXTINF:0.1,\nsegment.ts\n#EXT-X-ENDLIST\n", encoding="utf-8")
    return path


def write_transcript(path: Path) -> Path:
    path.write_text('[{"text":"hello","start_offset":0,"end_offset":0.1}]', encoding="utf-8")
    return path


def assert_redacted_config_error(result, *, raw_values: list[str], expected: list[str]) -> None:
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error:" in result.stderr
    assert "Traceback" not in result.stderr
    for value in raw_values:
        assert value not in result.stderr
    for value in expected:
        assert value in result.stderr


def test_default_missing_config_is_non_fatal_for_monitor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    calls = patch_monitor(monkeypatch)

    result = invoke(["monitor", "fixture.ts"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[0]["stream_type"] == "auto"
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.db_path is None


def test_monitor_config_env_and_cli_precedence_with_root_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_monitor(monkeypatch)
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config-private.db"

[monitor]
stream_type = "hls"
timeout_seconds = 8
""",
    )

    result = invoke(
        ["udp://239.1.1.1:5000", "--config", str(config_path), "--stream-type", "udp", "--timeout", "2.5"],
        env={"TIDEMARK_DB": "env-private.db", "TIDEMARK_MONITOR_TIMEOUT_SECONDS": "4"},
    )

    assert result.exit_code == 0, result.output
    assert calls[0] == {"source": "udp://239.1.1.1:5000", "stream_type": "udp", "timeout": 2.5}
    options = calls[1]["options"]
    assert isinstance(options, MonitorOptions)
    assert options.source_url == "udp://239.1.1.1:5000"
    assert options.timeout == 2.5
    assert options.db_path == Path("env-private.db")


def test_explicit_missing_config_fails_before_monitor_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_monitor(monkeypatch)
    missing = tmp_path / "secret-missing.toml"

    result = invoke(["monitor", "http://example.test/live.m3u8?token=secret", "--config", str(missing)])

    assert calls == []
    assert_redacted_config_error(result, raw_values=[str(tmp_path), "token=secret"], expected=["config", missing.name])


def test_bad_config_fails_with_field_source_and_redacted_value_before_monitor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_monitor(monkeypatch)
    config_path = write_config(tmp_path, "[monitor]\ntimeout_seconds = \"private-timeout\"")

    result = invoke(["monitor", "fixture.ts", "--config", str(config_path)])

    assert calls == []
    assert_redacted_config_error(
        result,
        raw_values=[str(tmp_path), "private-timeout", str(config_path)],
        expected=["monitor.timeout_seconds", "source=config", "[redacted-value]"],
    )


def test_bad_env_fails_with_field_source_and_no_monitor_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_monitor(monkeypatch)

    result = invoke(["monitor", "fixture.ts"], env={"TIDEMARK_MONITOR_TIMEOUT_SECONDS": "private-timeout"})

    assert calls == []
    assert_redacted_config_error(
        result,
        raw_values=["private-timeout"],
        expected=["monitor.timeout_seconds", "source=TIDEMARK_MONITOR_TIMEOUT_SECONDS"],
    )


def test_ingest_config_env_and_cli_precedence_with_redacted_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(monkeypatch)
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config.db"
retention_dir = "config-retained"

[ingest]
include_manifest_markers = false
fingerprint = false

[fingerprint]
api_key = "config-secret"
lookup_timeout_seconds = 9
""",
    )

    result = invoke(
        [
            "ingest",
            str(source),
            "--config",
            str(config_path),
            "--db",
            "cli.db",
            "--fixture-transcript",
            str(transcript),
            "--markers",
            "--fingerprint",
        ],
        env={"ACOUSTID_API_KEY": "env-secret", "TIDEMARK_LOOKUP_TIMEOUT_SECONDS": "4"},
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=1 words=0 markers=0 retained=0 songs=0 issues=0\n"
    assert "env-secret" not in result.stdout + result.stderr
    assert len(calls) == 1
    call = calls[0]
    assert call["db_path"] == Path("cli.db")
    assert call["include_manifest_markers"] is True
    assert call["fingerprint"] is True
    assert call["acoustid_api_key"] == "env-secret"
    assert call["lookup_timeout_seconds"] == 4.0
    assert call["retention_dir"] == Path("config-retained")


def test_ingest_explicit_false_cli_overrides_true_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(monkeypatch)
    config_path = write_config(
        tmp_path,
        """
[ingest]
include_manifest_markers = true
fingerprint = true
""",
    )

    result = invoke(
        ["ingest", str(source), "--config", str(config_path), "--fixture-transcript", str(transcript), "--no-markers", "--no-fingerprint"]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["include_manifest_markers"] is False
    assert calls[0]["fingerprint"] is False
    assert calls[0]["acoustid_api_key"] is None


def test_ingest_bad_secret_type_fails_redacted_before_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(monkeypatch)
    config_path = write_config(tmp_path, "[fingerprint]\napi_key = 12345")

    result = invoke(["ingest", str(source), "--config", str(config_path), "--fixture-transcript", str(transcript)])

    assert calls == []
    assert_redacted_config_error(
        result,
        raw_values=["12345", str(source), str(transcript), str(tmp_path)],
        expected=["fingerprint.api_key", "source=config", "[redacted]"],
    )
