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
        source_iter = marker_source() if callable(marker_source) else marker_source
        calls.append({"marker_source": source_iter, "options": options})
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


def patch_search(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_search_transcript_db(path, query: str, *, context_seconds: float = 5.0):
        calls.append({"path": Path(path), "query": query, "context_seconds": context_seconds})
        return ()

    monkeypatch.setattr("tidemark.cli.cmd_search.search_transcript_db", fake_search_transcript_db)
    return calls


def patch_report(monkeypatch: pytest.MonkeyPatch, command: str) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_plays_report_db(path, *, since_seconds=None, source_url=None, min_score: float = 0.8):
        calls.append(
            {
                "command": "plays",
                "path": Path(path),
                "since_seconds": since_seconds,
                "source_url": source_url,
                "min_score": min_score,
            }
        )
        return ()

    def fake_repeats_report_db(
        path,
        *,
        since_seconds=None,
        source_url=None,
        min_count: int = 2,
        min_score: float = 0.8,
    ):
        calls.append(
            {
                "command": "repeats",
                "path": Path(path),
                "since_seconds": since_seconds,
                "source_url": source_url,
                "min_count": min_count,
                "min_score": min_score,
            }
        )
        return ()

    def fake_ads_report_db(path, *, since_seconds=None, source_url=None):
        calls.append(
            {"command": "ads", "path": Path(path), "since_seconds": since_seconds, "source_url": source_url}
        )
        return ()

    if command == "plays":
        monkeypatch.setattr("tidemark.cli.cmd_report.plays_report_db", fake_plays_report_db)
    elif command == "repeats":
        monkeypatch.setattr("tidemark.cli.cmd_report.repeats_report_db", fake_repeats_report_db)
    elif command == "ads":
        monkeypatch.setattr("tidemark.cli.cmd_report.ads_report_db", fake_ads_report_db)
    else:
        raise AssertionError(f"unknown report command: {command}")
    return calls


def patch_clip(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_export_clip_db(path, *, at_seconds: float, context_seconds: float, out_path):
        from tidemark.clip import ClipExportResult

        calls.append(
            {
                "path": Path(path),
                "at_seconds": at_seconds,
                "context_seconds": context_seconds,
                "out_path": Path(out_path),
            }
        )
        return ClipExportResult(
            path=Path(out_path),
            start_ts=1.0,
            end_ts=2.0,
            duration_seconds=1.0,
            sample_rate=48_000,
            channels=2,
            sample_format="s16le",
            byte_length=100,
            sha256="a" * 64,
        )

    monkeypatch.setattr("tidemark.cli.cmd_clip.export_clip_db", fake_export_clip_db)
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


def test_search_config_env_and_cli_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_search(monkeypatch)
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config-search.db"

[search]
context_seconds = 9.5
""",
    )

    result = invoke(
        ["search", "needle", "--config", str(config_path), "--db", "cli-search.db", "--context", "1.25"],
        env={"TIDEMARK_DB": "env-search.db"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [{"path": Path("cli-search.db"), "query": "needle", "context_seconds": 1.25}]


def test_search_env_db_overrides_config_and_config_context_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = patch_search(monkeypatch)
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config-search.db"

[search]
context_seconds = 7
""",
    )

    result = invoke(["search", "needle", "--config", str(config_path)], env={"TIDEMARK_DB": "env-search.db"})

    assert result.exit_code == 0, result.output
    assert calls == [{"path": Path("env-search.db"), "query": "needle", "context_seconds": 7.0}]


def test_search_bad_config_fails_before_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_search(monkeypatch)
    config_path = write_config(tmp_path, "[search]\ncontext_seconds = \"private-context\"")

    result = invoke(["search", "needle", "--config", str(config_path)])

    assert calls == []
    assert_redacted_config_error(
        result,
        raw_values=["private-context", str(tmp_path), str(config_path)],
        expected=["search.context_seconds", "source=config", "[redacted-value]"],
    )


def test_report_config_env_and_cli_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_report(monkeypatch, "repeats")
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config-report.db"

[report]
min_score = 0.44
min_count = 4
""",
    )

    result = invoke(
        [
            "report",
            "repeats",
            "--config",
            str(config_path),
            "--db",
            "cli-report.db",
            "--min-score",
            "0.91",
            "--min-count",
            "8",
        ],
        env={"TIDEMARK_DB": "env-report.db"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "command": "repeats",
            "path": Path("cli-report.db"),
            "since_seconds": None,
            "source_url": None,
            "min_count": 8,
            "min_score": 0.91,
        }
    ]


def test_report_env_db_overrides_config_and_config_defaults_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = patch_report(monkeypatch, "plays")
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config-report.db"

[report]
min_score = 0.55
min_count = 6
""",
    )

    result = invoke(["report", "plays", "--config", str(config_path)], env={"TIDEMARK_DB": "env-report.db"})

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "command": "plays",
            "path": Path("env-report.db"),
            "since_seconds": None,
            "source_url": None,
            "min_score": 0.55,
        }
    ]


def test_report_repeats_uses_config_min_count_and_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_report(monkeypatch, "repeats")
    config_path = write_config(tmp_path, "[report]\nmin_score = 0.66\nmin_count = 5")

    result = invoke(["report", "repeats", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "command": "repeats",
            "path": Path("tidemark.db"),
            "since_seconds": None,
            "source_url": None,
            "min_count": 5,
            "min_score": 0.66,
        }
    ]


def test_report_ads_uses_shared_config_db_without_report_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = patch_report(monkeypatch, "ads")
    config_path = write_config(
        tmp_path,
        """
[paths]
db = "config-report.db"

[report]
min_score = 0.55
min_count = 6
""",
    )

    result = invoke(["report", "ads", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert calls == [
        {"command": "ads", "path": Path("config-report.db"), "since_seconds": None, "source_url": None}
    ]


def test_report_bad_config_fails_before_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_report(monkeypatch, "plays")
    config_path = write_config(tmp_path, "[report]\nmin_score = \"private-score\"")

    result = invoke(["report", "plays", "--config", str(config_path)])

    assert calls == []
    assert_redacted_config_error(
        result,
        raw_values=["private-score", str(tmp_path), str(config_path)],
        expected=["report.min_score", "source=config", "[redacted-value]"],
    )


def test_clip_config_env_and_cli_db_precedence_keeps_action_inputs_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = patch_clip(monkeypatch)
    out_path = tmp_path / "clip.wav"
    config_path = write_config(tmp_path, "[paths]\ndb = \"config-clip.db\"")

    result = invoke(
        [
            "clip",
            "--at",
            "12.5",
            "--context",
            "2",
            "--out",
            str(out_path),
            "--config",
            str(config_path),
            "--db",
            "cli-clip.db",
        ],
        env={"TIDEMARK_DB": "env-clip.db"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {"path": Path("cli-clip.db"), "at_seconds": 12.5, "context_seconds": 2.0, "out_path": out_path}
    ]


def test_clip_env_db_overrides_config_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_clip(monkeypatch)
    out_path = tmp_path / "clip.wav"
    config_path = write_config(tmp_path, "[paths]\ndb = \"config-clip.db\"")

    result = invoke(
        ["clip", "--at", "12.5", "--context", "2", "--out", str(out_path), "--config", str(config_path)],
        env={"TIDEMARK_DB": "env-clip.db"},
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {"path": Path("env-clip.db"), "at_seconds": 12.5, "context_seconds": 2.0, "out_path": out_path}
    ]


def test_clip_bad_config_fails_before_delegation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_clip(monkeypatch)
    out_path = tmp_path / "clip.wav"
    config_path = write_config(tmp_path, "[paths]\ndb = 12345")

    result = invoke(["clip", "--at", "12.5", "--context", "2", "--out", str(out_path), "--config", str(config_path)])

    assert calls == []
    assert_redacted_config_error(
        result,
        raw_values=["12345", str(tmp_path), str(config_path), str(out_path)],
        expected=["paths.db", "source=config", "[redacted-path]"],
    )


def test_clip_still_requires_explicit_at_and_out_even_with_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = patch_clip(monkeypatch)
    config_path = write_config(tmp_path, "[paths]\ndb = \"config-clip.db\"")

    result = invoke(["clip", "--context", "2", "--config", str(config_path)])

    assert result.exit_code != 0
    assert calls == []
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
