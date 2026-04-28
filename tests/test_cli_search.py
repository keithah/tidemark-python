from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.search import TranscriptDatabaseEmpty, TranscriptDatabaseMissing, TranscriptSearchResult


runner = CliRunner()


def invoke(args: list[str]):
    from tidemark.cli.main import app

    return runner.invoke(app, args)


@dataclass(frozen=True)
class SearchCall:
    path: Path
    query: str
    context_seconds: float


def make_result(**overrides: object) -> TranscriptSearchResult:
    values = {
        "source_url": "https://example.test/live.m3u8?token=secret",
        "segment_id": 7,
        "segment_sequence": 42,
        "hit_start_ts": 12.25,
        "hit_end_ts": 12.75,
        "context_start_ts": 10.0,
        "context_end_ts": 15.5,
        "context_text": "before hello world after",
        "matched_text": "hello world",
        "word_ids": (101, 102),
    }
    values.update(overrides)
    return TranscriptSearchResult(**values)  # type: ignore[arg-type]


def patch_search(monkeypatch: pytest.MonkeyPatch, results: tuple[TranscriptSearchResult, ...] = ()) -> list[SearchCall]:
    calls: list[SearchCall] = []

    def fake_search_transcript_db(path, query: str, *, context_seconds: float = 5.0):
        calls.append(SearchCall(Path(path), query, context_seconds))
        return results

    monkeypatch.setattr("tidemark.cli.cmd_search.search_transcript_db", fake_search_transcript_db)
    return calls


def test_search_command_delegates_to_library_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_search(monkeypatch, (make_result(),))

    result = invoke(["search", "hello world"])

    assert result.exit_code == 0, result.output
    assert calls == [SearchCall(Path("tidemark.db"), "hello world", 5.0)]
    assert result.stderr == ""


def test_root_alias_does_not_treat_search_as_monitor_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_search(monkeypatch)
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(url)

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["search", "hello"])

    assert result.exit_code == 0, result.output
    assert calls == [SearchCall(Path("tidemark.db"), "hello", 5.0)]
    assert monitor_calls == []


def test_search_command_passes_db_and_context_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_search(monkeypatch)
    db_path = tmp_path / "transcripts.sqlite"

    result = invoke(["search", "hello", "--db", str(db_path), "--context", "1.25"])

    assert result.exit_code == 0, result.output
    assert calls == [SearchCall(db_path, "hello", 1.25)]


def test_search_human_output_has_stable_one_line_per_result(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_search(
        monkeypatch,
        (
            make_result(),
            make_result(
                source_url="file:///tmp/sample.ts",
                segment_id=8,
                segment_sequence=43,
                hit_start_ts=98.0,
                hit_end_ts=98.4,
                context_start_ts=97.0,
                context_end_ts=99.0,
                context_text="another hit context",
                matched_text="hit",
                word_ids=(201,),
            ),
        ),
    )

    result = invoke(["search", "hello"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "https://example.test/live.m3u8?token=secret | 12.250s | segment 7 seq 42 | before hello world after\n"
        "file:///tmp/sample.ts | 98.000s | segment 8 seq 43 | another hit context\n"
    )


def test_search_json_output_uses_stable_snake_case_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_search(monkeypatch, (make_result(),))

    result = invoke(["search", "hello", "--json"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        '[{"source_url":"https://example.test/live.m3u8?token=secret",'
        '"segment_id":7,"segment_sequence":42,"hit_start_ts":12.25,'
        '"hit_end_ts":12.75,"context_start_ts":10.0,"context_end_ts":15.5,'
        '"context_text":"before hello world after","matched_text":"hello world",'
        '"word_ids":[101,102]}]\n'
    )


@pytest.mark.parametrize(
    ("args", "expected_stdout"),
    [
        (["search", "absent"], "No transcript matches found.\n"),
        (["search", "absent", "--json"], "[]\n"),
    ],
)
def test_no_matches_exit_zero_for_human_and_json(monkeypatch: pytest.MonkeyPatch, args: list[str], expected_stdout: str) -> None:
    calls = patch_search(monkeypatch)

    result = invoke(args)

    assert result.exit_code == 0, result.output
    assert result.stdout == expected_stdout
    assert result.stderr == ""
    assert len(calls) == 1


@pytest.mark.parametrize("args", [["search", "   "], ["search", "hello", "--context", "-0.01"]])
def test_malformed_inputs_are_rejected_before_search_starts(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    calls = patch_search(monkeypatch)

    result = invoke(args)

    assert result.exit_code != 0
    assert calls == []
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "error",
    [
        TranscriptDatabaseMissing("database path does not exist"),
        TranscriptDatabaseEmpty("transcript_words table is empty"),
    ],
)
def test_expected_library_errors_are_redacted_and_exit_one(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fake_search_transcript_db(path, query: str, *, context_seconds: float = 5.0):
        raise error

    monkeypatch.setattr("tidemark.cli.cmd_search.search_transcript_db", fake_search_transcript_db)

    result = invoke(["search", "private query", "--db", "/tmp/private.sqlite"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert f"[tidemark] error: {error}" in result.stderr
    assert "private query" not in result.stderr
    assert "/tmp/private.sqlite" not in result.stderr
    assert "Traceback" not in result.stderr


def test_unexpected_library_errors_are_generic_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search_transcript_db(path, query: str, *, context_seconds: float = 5.0):
        raise RuntimeError("boom private query /tmp/private.sqlite")

    monkeypatch.setattr("tidemark.cli.cmd_search.search_transcript_db", fake_search_transcript_db)

    result = invoke(["search", "private query", "--db", "/tmp/private.sqlite"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: search failed" in result.stderr
    assert "private query" not in result.stderr
    assert "/tmp/private.sqlite" not in result.stderr
    assert "boom" not in result.stderr
    assert "Traceback" not in result.stderr


def test_search_help_does_not_invoke_monitor_or_search(monkeypatch: pytest.MonkeyPatch) -> None:
    search_calls = patch_search(monkeypatch)
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(url)

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["search", "--help"])

    assert result.exit_code == 0
    assert "search" in result.stdout.lower()
    assert search_calls == []
    assert monitor_calls == []
