from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.ingest.pipeline import IngestPipelineResult


runner = CliRunner()


def invoke(args: list[str]):
    from tidemark.cli.main import app

    return runner.invoke(app, args)


@dataclass(frozen=True)
class IngestCall:
    source: Path
    db_path: Path
    fixture_words: tuple[tuple[str, float, float, float | None], ...]
    language: str | None
    engine: str | None
    source_url: str | None
    include_manifest_markers: bool


def write_manifest(path: Path, segment_name: str = "segment.ts") -> Path:
    (path.parent / segment_name).write_bytes(b"not decoded by CLI delegation tests")
    path.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-MEDIA-SEQUENCE:7",
                "#EXT-X-CUE-OUT:DURATION=15.0",
                "#EXTINF:0.20,",
                segment_name,
                "#EXT-X-CUE-IN",
                "#EXT-X-ENDLIST",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_transcript(path: Path, words: list[dict[str, object]] | None = None) -> Path:
    path.write_text(
        json.dumps(
            words
            if words is not None
            else [
                {"text": "hello", "start_offset": 0.0, "end_offset": 0.1, "confidence": 0.9},
                {"text": "tidemark", "start_offset": 0.2, "end_offset": 0.3},
                {"text": "search", "start_offset": 0.4, "end_offset": 0.5, "confidence": None},
            ]
        ),
        encoding="utf-8",
    )
    return path


def patch_ingest(monkeypatch: pytest.MonkeyPatch, result: IngestPipelineResult) -> list[IngestCall]:
    calls: list[IngestCall] = []

    def fake_ingest_source_to_db(
        source,
        *,
        db_path,
        transcriber,
        source_url: str | None = None,
        include_manifest_markers: bool = True,
    ):
        calls.append(
            IngestCall(
                source=Path(source),
                db_path=Path(db_path),
                fixture_words=tuple(transcriber.fixture_words),
                language=transcriber.language,
                engine=transcriber.engine,
                source_url=source_url,
                include_manifest_markers=include_manifest_markers,
            )
        )
        return result

    monkeypatch.setattr("tidemark.cli.cmd_ingest.ingest_source_to_db", fake_ingest_source_to_db)
    return calls


def test_ingest_command_delegates_once_and_prints_safe_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "private-playlist.m3u8")
    transcript = write_transcript(tmp_path / "private-transcript.json")
    db_path = tmp_path / "tidemark.sqlite3"
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101, 102, 103), ad_event_ids=(5,), issues=()),
    )

    result = invoke(
        [
            "ingest",
            str(source),
            "--db",
            str(db_path),
            "--fixture-transcript",
            str(transcript),
            "--source-url",
            "fixture://integration/private.m3u8?token=secret",
        ]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=1 words=3 markers=1 issues=0\n"
    assert result.stderr == ""
    assert calls == [
        IngestCall(
            source=source,
            db_path=db_path,
            fixture_words=(
                ("hello", 0.0, 0.1, 0.9),
                ("tidemark", 0.2, 0.3, None),
                ("search", 0.4, 0.5, None),
            ),
            language="en",
            engine="deterministic-fixture",
            source_url="fixture://integration/private.m3u8?token=secret",
            include_manifest_markers=True,
        )
    ]
    assert "token=secret" not in result.stdout
    assert str(source) not in result.stdout
    assert str(transcript) not in result.stdout


def test_ingest_reports_recoverable_pipeline_issues_as_counts_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(
            segment_ids=(11,),
            transcript_word_ids=(),
            ad_event_ids=(5,),
            issues=(object(), object()),  # type: ignore[arg-type]
        ),
    )

    result = invoke(["ingest", str(source), "--fixture-transcript", str(transcript)])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=1 words=0 markers=1 issues=2\n"
    assert result.stderr == ""
    assert len(calls) == 1


def test_no_markers_disables_manifest_marker_persistence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )

    result = invoke(["ingest", str(source), "--fixture-transcript", str(transcript), "--no-markers"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=1 words=1 markers=0 issues=0\n"
    assert calls[0].include_manifest_markers is False


@pytest.mark.parametrize(
    ("words", "expected_field"),
    [
        ([{"text": "", "start_offset": 0.0, "end_offset": 0.1}], "text"),
        ([{"text": "private transcript", "start_offset": -1.0, "end_offset": 0.1}], "start_offset"),
        ([{"text": "private transcript", "start_offset": 0.0, "end_offset": 0.1, "confidence": 2.0}], "confidence"),
    ],
)
def test_malformed_fixture_transcript_is_redacted_and_rejected_before_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, words: list[dict[str, object]], expected_field: str
) -> None:
    source = write_manifest(tmp_path / "private-source.m3u8")
    transcript = write_transcript(tmp_path / "private-transcript.json", words)
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )

    result = invoke(
        [
            "ingest",
            str(source),
            "--fixture-transcript",
            str(transcript),
            "--source-url",
            "fixture://private.example/live.m3u8?token=secret",
        ]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert f"[tidemark] error: {expected_field}" in result.stderr
    assert calls == []
    assert "Traceback" not in result.stderr
    assert "private transcript" not in result.stderr
    assert "token=secret" not in result.stderr
    assert str(source) not in result.stderr
    assert str(transcript) not in result.stderr
    assert source.name not in result.stderr
    assert transcript.name not in result.stderr


def test_invalid_fixture_json_is_redacted_and_rejected_before_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_manifest(tmp_path / "private-source.m3u8")
    transcript = tmp_path / "private-transcript.json"
    transcript.write_text("private transcript is not json", encoding="utf-8")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )

    result = invoke(["ingest", str(source), "--fixture-transcript", str(transcript)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: json" in result.stderr
    assert calls == []
    assert "private transcript" not in result.stderr
    assert str(transcript) not in result.stderr
    assert "Traceback" not in result.stderr


def test_pipeline_setup_errors_are_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "private-source.m3u8"
    transcript = write_transcript(tmp_path / "private-transcript.json")

    def fake_ingest_source_to_db(*args, **kwargs):
        raise ValueError("private setup failed for /tmp/private-source.m3u8 token=secret")

    monkeypatch.setattr("tidemark.cli.cmd_ingest.ingest_source_to_db", fake_ingest_source_to_db)

    result = invoke(
        [
            "ingest",
            str(source),
            "--fixture-transcript",
            str(transcript),
            "--source-url",
            "fixture://private.example/live.m3u8?token=secret",
        ]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: ingest failed" in result.stderr
    assert "private" not in result.stderr
    assert "token=secret" not in result.stderr
    assert str(source) not in result.stderr
    assert str(transcript) not in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_fixture_transcript_is_a_cli_usage_error_before_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )

    result = invoke(["ingest", str(source)])

    assert result.exit_code != 0
    assert calls == []
    assert "--fixture-transcript" in result.stderr
    assert "Traceback" not in result.stderr


def test_ingest_is_registered_as_real_command_not_root_monitor_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(str(url))

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["ingest", str(source), "--fixture-transcript", str(transcript)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert monitor_calls == []


def test_ingest_help_lists_required_fixture_and_db_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(str(url))

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["ingest", "--help"])

    assert result.exit_code == 0
    assert "--db" in result.stdout
    assert "--fixture-transcript" in result.stdout
    assert monitor_calls == []


def test_unknown_ingest_option_fails_as_usage_error_not_monitor_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(str(url))

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["ingest", str(source), "--fixture-transcript", str(transcript), "--unknown"])

    assert result.exit_code != 0
    assert monitor_calls == []
    assert "Traceback" not in result.stderr
