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
    transcriber_is_none: bool
    fixture_words: tuple[tuple[str, float, float, float | None], ...]
    language: str | None
    engine: str | None
    source_url: str | None
    include_manifest_markers: bool
    fingerprint: bool
    acoustid_api_key: str | None
    lookup_timeout_seconds: float | None
    retention_dir: Path | None


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
        fingerprint: bool = False,
        acoustid_api_key: str | None = None,
        lookup_timeout_seconds: float | None = None,
        retention_dir: str | Path | None = None,
        **kwargs,
    ):
        calls.append(
            IngestCall(
                source=Path(source),
                db_path=Path(db_path),
                transcriber_is_none=transcriber is None,
                fixture_words=tuple(transcriber.fixture_words) if transcriber is not None else (),
                language=transcriber.language if transcriber is not None else None,
                engine=transcriber.engine if transcriber is not None else None,
                source_url=source_url,
                include_manifest_markers=include_manifest_markers,
                fingerprint=fingerprint,
                acoustid_api_key=acoustid_api_key,
                lookup_timeout_seconds=lookup_timeout_seconds,
                retention_dir=Path(retention_dir) if retention_dir is not None else None,
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
            transcriber_is_none=False,
            fixture_words=(
                ("hello", 0.0, 0.1, 0.9),
                ("tidemark", 0.2, 0.3, None),
                ("search", 0.4, 0.5, None),
            ),
            language="en",
            engine="deterministic-fixture",
            source_url="fixture://integration/private.m3u8?token=secret",
            include_manifest_markers=True,
            fingerprint=False,
            acoustid_api_key=None,
            lookup_timeout_seconds=None,
            retention_dir=None,
        )
    ]
    assert "retained=" not in result.stdout
    assert "songs=" not in result.stdout
    assert "token=secret" not in result.stdout
    assert str(source) not in result.stdout
    assert str(transcript) not in result.stdout


def test_fingerprint_output_includes_retained_and_song_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(
            segment_ids=(11, 12),
            transcript_word_ids=(),
            ad_event_ids=(5,),
            retained_audio_ids=(20, 21),
            song_ids=(30,),
            issues=(object(),),  # type: ignore[arg-type]
        ),
    )
    monkeypatch.setenv("ACOUSTID_API_KEY", "private-api-key")

    result = invoke(["ingest", str(source), "--fingerprint"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=2 words=0 markers=1 retained=2 songs=1 issues=1\n"
    assert result.stderr == ""
    assert len(calls) == 1
    assert calls[0].transcriber_is_none is True
    assert calls[0].fingerprint is True
    assert calls[0].acoustid_api_key == "private-api-key"
    assert calls[0].lookup_timeout_seconds is None
    assert calls[0].retention_dir is None
    assert "private-api-key" not in result.stdout


def test_fingerprint_ingest_without_fixture_delegates_with_no_transcriber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(), ad_event_ids=(), issues=()),
    )
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)

    result = invoke(["ingest", str(source), "--fingerprint"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=1 words=0 markers=0 retained=0 songs=0 issues=0\n"
    assert calls == [
        IngestCall(
            source=source,
            db_path=Path("tidemark.db"),
            transcriber_is_none=True,
            fixture_words=(),
            language=None,
            engine=None,
            source_url=None,
            include_manifest_markers=True,
            fingerprint=True,
            acoustid_api_key=None,
            lookup_timeout_seconds=None,
            retention_dir=None,
        )
    ]


def test_fingerprint_ingest_with_fixture_delegates_with_deterministic_transcriber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_manifest(tmp_path / "playlist.m3u8")
    transcript = write_transcript(tmp_path / "transcript.json")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )

    result = invoke(["ingest", str(source), "--fingerprint", "--fixture-transcript", str(transcript)])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Ingest complete: segments=1 words=1 markers=0 retained=0 songs=0 issues=0\n"
    assert len(calls) == 1
    assert calls[0].transcriber_is_none is False
    assert calls[0].fixture_words == (
        ("hello", 0.0, 0.1, 0.9),
        ("tidemark", 0.2, 0.3, None),
        ("search", 0.4, 0.5, None),
    )
    assert calls[0].language == "en"
    assert calls[0].engine == "deterministic-fixture"
    assert calls[0].fingerprint is True


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
    assert calls[0].fingerprint is False


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
    assert calls[0].fingerprint is False


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
    source = write_manifest(tmp_path / "private-playlist.m3u8")
    calls = patch_ingest(
        monkeypatch,
        IngestPipelineResult(segment_ids=(11,), transcript_word_ids=(101,), ad_event_ids=(), issues=()),
    )

    result = invoke(["ingest", str(source)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert calls == []
    assert "--fixture-transcript is required unless --fingerprint is enabled" in result.stderr
    assert "Traceback" not in result.stderr
    assert str(source) not in result.stderr
    assert source.name not in result.stderr


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


def test_ingest_help_lists_required_fixture_db_and_fingerprint_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(str(url))

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["ingest", "--help"])

    assert result.exit_code == 0
    assert "--db" in result.stdout
    assert "--fixture-transcript" in result.stdout
    assert "--fingerprint" in result.stdout
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
    assert "No such option" in result.stderr
    assert "--unknown" in result.stderr
    assert "Traceback" not in result.stderr
