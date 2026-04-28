from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tidemark.clip import ClipCoverageMissing, ClipExportResult


runner = CliRunner()


def invoke(args: list[str]):
    from tidemark.cli.main import app

    return runner.invoke(app, args)


@dataclass(frozen=True)
class ClipCall:
    path: Path
    at_seconds: float
    context_seconds: float
    out_path: Path


def make_result(**overrides: object) -> ClipExportResult:
    values = {
        "path": Path("/private/output/clip.wav"),
        "start_ts": 10.0,
        "end_ts": 12.5,
        "duration_seconds": 2.5,
        "sample_rate": 48_000,
        "channels": 2,
        "sample_format": "s16le",
        "byte_length": 1234,
        "sha256": "a" * 64,
    }
    values.update(overrides)
    return ClipExportResult(**values)  # type: ignore[arg-type]


def patch_clip(monkeypatch: pytest.MonkeyPatch, result: ClipExportResult | None = None) -> list[ClipCall]:
    calls: list[ClipCall] = []

    def fake_export_clip_db(path, *, at_seconds: float, context_seconds: float, out_path):
        calls.append(ClipCall(Path(path), at_seconds, context_seconds, Path(out_path)))
        return result or make_result()

    monkeypatch.setattr("tidemark.cli.cmd_clip.export_clip_db", fake_export_clip_db)
    return calls


def test_clip_command_delegates_to_library_with_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_clip(monkeypatch)
    out_path = tmp_path / "clip.wav"

    result = invoke(["clip", "--at", "12.5", "--context", "2.0", "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    assert calls == [ClipCall(Path("tidemark.db"), 12.5, 2.0, out_path)]
    assert result.stderr == ""


def test_root_alias_does_not_treat_clip_as_monitor_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_clip(monkeypatch)
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(url)

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["clip", "--at", "1", "--context", "1", "--out", str(tmp_path / "clip.wav")])

    assert result.exit_code == 0, result.output
    assert calls == [ClipCall(Path("tidemark.db"), 1.0, 1.0, tmp_path / "clip.wav")]
    assert monitor_calls == []


def test_clip_command_passes_db_and_out_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_clip(monkeypatch)
    db_path = tmp_path / "private.sqlite3"
    out_path = tmp_path / "exported.wav"

    result = invoke(["clip", "--at", "20.25", "--context", "3.5", "--db", str(db_path), "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    assert calls == [ClipCall(db_path, 20.25, 3.5, out_path)]


def test_clip_success_output_is_stable_metadata_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    private_out = tmp_path / "private-fixture-name.wav"
    patch_clip(
        monkeypatch,
        make_result(path=private_out, start_ts=98.125, duration_seconds=4.25, byte_length=4096, sha256="b" * 64),
    )

    result = invoke(
        [
            "clip",
            "--at",
            "100",
            "--context",
            "2",
            "--db",
            str(tmp_path / "private-db.sqlite3"),
            "--out",
            str(private_out),
        ]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == f"Clip exported: start=98.125 duration=4.250 bytes=4096 sha256={'b' * 64}\n"
    assert str(private_out) not in result.stdout
    assert str(tmp_path) not in result.stdout
    assert "private" not in result.stdout
    assert result.stderr == ""


def test_missing_required_out_fails_before_library_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_clip(monkeypatch)

    result = invoke(["clip", "--at", "12", "--context", "2"])

    assert result.exit_code != 0
    assert calls == []
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("args", [["clip", "--at", "-0.01", "--context", "1"], ["clip", "--at", "1", "--context", "-0.01"]])
def test_malformed_numeric_inputs_are_rejected_before_library_delegation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
) -> None:
    calls = patch_clip(monkeypatch)

    result = invoke([*args, "--out", str(tmp_path / "clip.wav")])

    assert result.exit_code != 0
    assert calls == []
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_expected_library_errors_are_redacted_and_exit_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    error = ClipCoverageMissing("no retained audio covers timestamp")

    def fake_export_clip_db(path, *, at_seconds: float, context_seconds: float, out_path):
        raise error

    monkeypatch.setattr("tidemark.cli.cmd_clip.export_clip_db", fake_export_clip_db)

    result = invoke(
        [
            "clip",
            "--at",
            "12",
            "--context",
            "2",
            "--db",
            str(tmp_path / "private.sqlite3"),
            "--out",
            str(tmp_path / "secret-clip.wav"),
        ]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: no retained audio covers timestamp" in result.stderr
    assert str(tmp_path) not in result.stderr
    assert "private.sqlite3" not in result.stderr
    assert "secret-clip.wav" not in result.stderr
    assert "Traceback" not in result.stderr


def test_unexpected_library_errors_are_generic_and_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_export_clip_db(path, *, at_seconds: float, context_seconds: float, out_path):
        raise RuntimeError(f"boom {path} {out_path}")

    monkeypatch.setattr("tidemark.cli.cmd_clip.export_clip_db", fake_export_clip_db)

    result = invoke(
        [
            "clip",
            "--at",
            "12",
            "--context",
            "2",
            "--db",
            str(tmp_path / "private.sqlite3"),
            "--out",
            str(tmp_path / "secret-clip.wav"),
        ]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[tidemark] error: clip export failed" in result.stderr
    assert str(tmp_path) not in result.stderr
    assert "private.sqlite3" not in result.stderr
    assert "secret-clip.wav" not in result.stderr
    assert "boom" not in result.stderr
    assert "Traceback" not in result.stderr


def test_clip_help_does_not_invoke_monitor_or_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_clip(monkeypatch)
    monitor_calls: list[str] = []

    def fake_monitor(url, *args, **kwargs):
        monitor_calls.append(url)

    monkeypatch.setattr("tidemark.cli.main.monitor", fake_monitor)

    result = invoke(["clip", "--help"])

    assert result.exit_code == 0
    assert "clip" in result.stdout.lower()
    assert calls == []
    assert monitor_calls == []
