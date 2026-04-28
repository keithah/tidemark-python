from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.reports import AdSummaryReportRow, PlayReportRow, RepeatReportRow, ReportDatabaseMissing


runner = CliRunner()


def invoke(args: list[str]):
    from tidemark.cli.main import app

    return runner.invoke(app, args)


@dataclass(frozen=True)
class PlaysCall:
    path: Path
    since_seconds: float | None
    source_url: str | None
    min_score: float


@dataclass(frozen=True)
class RepeatsCall:
    path: Path
    since_seconds: float | None
    source_url: str | None
    min_count: int
    min_score: float


@dataclass(frozen=True)
class AdsCall:
    path: Path
    since_seconds: float | None
    source_url: str | None


def make_play(**overrides: object) -> PlayReportRow:
    values = {
        "song_id": 11,
        "source_url": "https://example.test/live.m3u8?token=secret",
        "segment_id": 7,
        "segment_sequence": 42,
        "start_ts": 12.25,
        "duration_seconds": 4.5,
        "title": "Needle Song",
        "artist": "Needle Artist",
        "album": "Needle Album",
        "score": 0.93456,
        "acoustid_id": "acoustid-secret",
        "recording_id": "recording-secret",
        "lookup_source": "fixture",
    }
    values.update(overrides)
    return PlayReportRow(**values)  # type: ignore[arg-type]


def make_repeat(**overrides: object) -> RepeatReportRow:
    values = {
        "identity": "recording:recording-secret",
        "title": "Needle Song",
        "artist": "Needle Artist",
        "album": "Needle Album",
        "count": 2,
        "first_start_ts": 12.25,
        "last_start_ts": 72.5,
        "source_urls": ("https://example.test/live.m3u8?token=secret", "fixture://stream-b"),
        "song_ids": (11, 12),
        "best_score": 0.93456,
        "acoustid_id": "acoustid-secret",
        "recording_id": "recording-secret",
    }
    values.update(overrides)
    return RepeatReportRow(**values)  # type: ignore[arg-type]


def make_ad(**overrides: object) -> AdSummaryReportRow:
    values = {
        "source_url": "https://example.test/live.m3u8?token=secret",
        "classification": "BREAK_START",
        "marker_type": "SCTE35",
        "count": 3,
        "first_ts": 10.0,
        "last_ts": 50.5,
        "total_break_duration": 45.25,
    }
    values.update(overrides)
    return AdSummaryReportRow(**values)  # type: ignore[arg-type]


def patch_plays(monkeypatch: pytest.MonkeyPatch, results: tuple[PlayReportRow, ...] = ()) -> list[PlaysCall]:
    calls: list[PlaysCall] = []

    def fake_plays_report_db(
        path,
        *,
        since_seconds: float | None = None,
        source_url: str | None = None,
        min_score: float = 0.8,
    ):
        calls.append(PlaysCall(Path(path), since_seconds, source_url, min_score))
        return results

    monkeypatch.setattr("tidemark.cli.cmd_report.plays_report_db", fake_plays_report_db)
    return calls


def patch_repeats(monkeypatch: pytest.MonkeyPatch, results: tuple[RepeatReportRow, ...] = ()) -> list[RepeatsCall]:
    calls: list[RepeatsCall] = []

    def fake_repeats_report_db(
        path,
        *,
        since_seconds: float | None = None,
        source_url: str | None = None,
        min_count: int = 2,
        min_score: float = 0.8,
    ):
        calls.append(RepeatsCall(Path(path), since_seconds, source_url, min_count, min_score))
        return results

    monkeypatch.setattr("tidemark.cli.cmd_report.repeats_report_db", fake_repeats_report_db)
    return calls


def patch_ads(monkeypatch: pytest.MonkeyPatch, results: tuple[AdSummaryReportRow, ...] = ()) -> list[AdsCall]:
    calls: list[AdsCall] = []

    def fake_ads_report_db(path, *, since_seconds: float | None = None, source_url: str | None = None):
        calls.append(AdsCall(Path(path), since_seconds, source_url))
        return results

    monkeypatch.setattr("tidemark.cli.cmd_report.ads_report_db", fake_ads_report_db)
    return calls


def test_report_plays_delegates_to_library_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_plays(monkeypatch, (make_play(),))

    result = invoke(["report", "plays"])

    assert result.exit_code == 0, result.output
    assert calls == [PlaysCall(Path("tidemark.db"), None, None, 0.8)]
    assert result.stderr == ""


def test_report_repeats_delegates_to_library_with_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_repeats(monkeypatch)
    db_path = tmp_path / "reports.sqlite"

    result = invoke(
        [
            "report",
            "repeats",
            "--db",
            str(db_path),
            "--since",
            "12.5",
            "--source",
            "fixture://stream-a",
            "--min-count",
            "3",
            "--min-score",
            "0.91",
        ]
    )

    assert result.exit_code == 0, result.output
    assert calls == [RepeatsCall(db_path, 12.5, "fixture://stream-a", 3, 0.91)]


def test_report_ads_delegates_to_library_with_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_ads(monkeypatch)
    db_path = tmp_path / "reports.sqlite"

    result = invoke(["report", "ads", "--db", str(db_path), "--since", "1.25", "--source", "fixture://stream-a"])

    assert result.exit_code == 0, result.output
    assert calls == [AdsCall(db_path, 1.25, "fixture://stream-a")]


def test_root_alias_does_not_treat_report_as_monitor_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_plays(monkeypatch)
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(url)

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["report", "plays"])

    assert result.exit_code == 0, result.output
    assert calls == [PlaysCall(Path("tidemark.db"), None, None, 0.8)]
    assert monitor_calls == []


def test_report_plays_human_output_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_plays(monkeypatch, (make_play(), make_play(source_url="fixture://stream-b", start_ts=98.0, duration_seconds=2.25, title="No Artist", artist=None, score=0.8)))

    result = invoke(["report", "plays"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "https://example.test/live.m3u8?token=secret | 12.250s-16.750s | "
        "segment 7 seq 42 | Needle Song | artist Needle Artist | score 0.935\n"
        "fixture://stream-b | 98.000s-100.250s | segment 7 seq 42 | No Artist | score 0.800\n"
    )


def test_report_repeats_human_output_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_repeats(monkeypatch, (make_repeat(),))

    result = invoke(["report", "repeats"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "Needle Song | artist Needle Artist | identity recording:recording-secret | count 2 | "
        "first 12.250s | last 72.500s | sources fixture://stream-b, "
        "https://example.test/live.m3u8?token=secret | best_score 0.935\n"
    )


def test_report_ads_human_output_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ads(monkeypatch, (make_ad(),))

    result = invoke(["report", "ads"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "https://example.test/live.m3u8?token=secret | BREAK_START | SCTE35 | count 3 | "
        "first 10.000s | last 50.500s | total_break_duration 45.250s\n"
    )


@pytest.mark.parametrize(
    ("args", "expected_stdout"),
    [
        (["report", "plays", "--json"], '[{"song_id":11,"source_url":"https://example.test/live.m3u8?token=secret","segment_id":7,"segment_sequence":42,"start_ts":12.25,"duration_seconds":4.5,"title":"Needle Song","artist":"Needle Artist","album":"Needle Album","score":0.93456,"acoustid_id":"acoustid-secret","recording_id":"recording-secret","lookup_source":"fixture"}]\n'),
        (["report", "repeats", "--json"], '[{"identity":"recording:recording-secret","title":"Needle Song","artist":"Needle Artist","album":"Needle Album","count":2,"first_start_ts":12.25,"last_start_ts":72.5,"source_urls":["https://example.test/live.m3u8?token=secret","fixture://stream-b"],"song_ids":[11,12],"best_score":0.93456,"acoustid_id":"acoustid-secret","recording_id":"recording-secret"}]\n'),
        (["report", "ads", "--json"], '[{"source_url":"https://example.test/live.m3u8?token=secret","classification":"BREAK_START","marker_type":"SCTE35","count":3,"first_ts":10.0,"last_ts":50.5,"total_break_duration":45.25}]\n'),
    ],
)
def test_report_json_output_uses_compact_dataclass_dicts(monkeypatch: pytest.MonkeyPatch, args: list[str], expected_stdout: str) -> None:
    patch_plays(monkeypatch, (make_play(),))
    patch_repeats(monkeypatch, (make_repeat(),))
    patch_ads(monkeypatch, (make_ad(),))

    result = invoke(args)

    assert result.exit_code == 0, result.output
    assert result.stdout == expected_stdout


@pytest.mark.parametrize(
    ("args", "expected_stdout"),
    [
        (["report", "plays"], "No identified plays found.\n"),
        (["report", "plays", "--json"], "[]\n"),
        (["report", "repeats"], "No repeat plays found.\n"),
        (["report", "repeats", "--json"], "[]\n"),
        (["report", "ads"], "No ad summaries found.\n"),
        (["report", "ads", "--json"], "[]\n"),
    ],
)
def test_no_report_rows_exit_zero_for_human_and_json(monkeypatch: pytest.MonkeyPatch, args: list[str], expected_stdout: str) -> None:
    plays_calls = patch_plays(monkeypatch)
    repeats_calls = patch_repeats(monkeypatch)
    ads_calls = patch_ads(monkeypatch)

    result = invoke(args)

    assert result.exit_code == 0, result.output
    assert result.stdout == expected_stdout
    assert result.stderr == ""
    assert len(plays_calls) + len(repeats_calls) + len(ads_calls) == 1


@pytest.mark.parametrize(
    "args",
    [
        ["report", "plays", "--since", "-0.01"],
        ["report", "plays", "--min-score", "-0.1"],
        ["report", "plays", "--min-score", "1.1"],
        ["report", "repeats", "--min-count", "0"],
    ],
)
def test_malformed_inputs_are_rejected_before_report_starts(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    plays_calls = patch_plays(monkeypatch)
    repeats_calls = patch_repeats(monkeypatch)
    ads_calls = patch_ads(monkeypatch)

    result = invoke(args)

    assert result.exit_code != 0
    assert plays_calls == []
    assert repeats_calls == []
    assert ads_calls == []
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_expected_report_errors_are_redacted_and_exit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ReportDatabaseMissing("database path does not exist")

    def fake_plays_report_db(path, *, since_seconds=None, source_url=None, min_score=0.8):
        raise error

    monkeypatch.setattr("tidemark.cli.cmd_report.plays_report_db", fake_plays_report_db)

    result = invoke(["report", "plays", "--db", "/tmp/private.sqlite", "--source", "https://secret.test/live"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert f"[tidemark] error: {error}" in result.stderr
    assert "/tmp/private.sqlite" not in result.stderr
    assert "https://secret.test/live" not in result.stderr
    assert "Traceback" not in result.stderr


def test_unexpected_report_errors_are_generic_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_repeats_report_db(path, *, since_seconds=None, source_url=None, min_count=2, min_score=0.8):
        raise RuntimeError("boom /tmp/private.sqlite secret title fingerprint")

    monkeypatch.setattr("tidemark.cli.cmd_report.repeats_report_db", fake_repeats_report_db)

    result = invoke(["report", "repeats", "--db", "/tmp/private.sqlite", "--source", "https://secret.test/live"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: report failed" in result.stderr
    assert "/tmp/private.sqlite" not in result.stderr
    assert "https://secret.test/live" not in result.stderr
    assert "boom" not in result.stderr
    assert "fingerprint" not in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("args", [["report"], ["report", "--help"], ["report", "plays", "--help"]])
def test_report_help_does_not_invoke_monitor_or_report(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    plays_calls = patch_plays(monkeypatch)
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(url)

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(args)

    assert result.exit_code == 0
    assert "report" in result.stdout.lower()
    assert plays_calls == []
    assert monitor_calls == []
