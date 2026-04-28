"""Deterministic ingest command implementation for the tidemark Typer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tidemark.config import ConfigError, IngestOverrides, default_runtime_dir, load_config, resolve_ingest_options
from tidemark.ingest import SegmentIngestError
from tidemark.ingest.pipeline import (
    IngestPipelineProgress,
    IngestPipelineResult,
    TranscriptFixtureError,
    ingest_source_to_db,
    load_fixture_transcript,
)
from tidemark.runtime.health import HealthReporter, create_reporter
from tidemark.transcribe import DeterministicTranscriber


SourceArgument = Annotated[
    Path,
    typer.Argument(help="Local manifest or media source to ingest."),
]
DbOption = Annotated[
    Path | None,
    typer.Option("--db", help="SQLite database to create or update."),
]
FixtureTranscriptOption = Annotated[
    Path | None,
    typer.Option("--fixture-transcript", help="Deterministic transcript JSON fixture for M002 ingest proof."),
]
FingerprintOption = Annotated[
    bool | None,
    typer.Option("--fingerprint/--no-fingerprint", help="Opt in to retained-audio fingerprinting and song identification."),
]
SourceUrlOption = Annotated[
    str | None,
    typer.Option("--source-url", help="Optional logical source URL stored with segment/transcript rows."),
]
MarkersOption = Annotated[
    bool | None,
    typer.Option("--markers/--no-markers", help="Enable or disable manifest marker persistence."),
]
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="TOML config file to load for command defaults."),
]


def run_ingest_command(
    source: Path,
    *,
    db_path: Path | None = None,
    fixture_transcript: Path | None = None,
    source_url: str | None = None,
    include_manifest_markers: bool | None = None,
    fingerprint: bool | None = None,
    config_path: Path | None = None,
) -> None:
    """Load optional deterministic fixtures, delegate once to the pipeline, and print safe counts."""
    try:
        config = load_config(config_path, explicit=config_path is not None)
        resolved = resolve_ingest_options(
            config,
            IngestOverrides(
                db_path=db_path,
                include_manifest_markers=include_manifest_markers,
                fingerprint=fingerprint,
            ),
        )
    except ConfigError as exc:
        _fatal(str(exc))

    reporter = _create_ingest_reporter(
        Path(config.paths.runtime_dir or default_runtime_dir()).expanduser(),
        source=source_url or source,
    )
    _report_start(reporter)

    fixture_words = None
    if fixture_transcript is not None:
        try:
            fixture_words = load_fixture_transcript(fixture_transcript)
        except TranscriptFixtureError as exc:
            _report_fail(reporter, str(exc), reason="fixture_error")
            _fatal(str(exc))
        except Exception:
            _report_fail(reporter, "fixture transcript could not be loaded", reason="fixture_error")
            _fatal("fixture transcript could not be loaded")
    elif not resolved.fingerprint:
        _report_fail(
            reporter,
            "--fixture-transcript is required unless --fingerprint is enabled",
            reason="fixture_error",
        )
        _fatal("--fixture-transcript is required unless --fingerprint is enabled")

    transcriber = None
    if fixture_words is not None:
        transcriber = DeterministicTranscriber(
            fixture_words,
            language="en",
            engine="deterministic-fixture",
        )

    try:
        result = ingest_source_to_db(
            source,
            db_path=resolved.db_path,
            transcriber=transcriber,
            source_url=source_url,
            include_manifest_markers=resolved.include_manifest_markers,
            fingerprint=resolved.fingerprint,
            acoustid_api_key=resolved.acoustid_api_key if resolved.fingerprint else None,
            lookup_timeout_seconds=resolved.lookup_timeout_seconds,
            retention_dir=resolved.retention_dir,
            progress_callback=_ingest_progress_callback(reporter),
        )
    except SegmentIngestError as exc:
        _report_fail(reporter, str(exc), reason="source_error")
        _fatal(str(exc))
    except Exception as exc:
        _report_fail(reporter, str(exc), reason="pipeline_error")
        _fatal("ingest failed")

    final_counters = _result_counters(result)
    _report_finish(reporter, counters=final_counters)
    output = (
        "Ingest complete: "
        f"segments={len(result.segment_ids)} "
        f"words={len(result.transcript_word_ids)} "
        f"markers={len(result.ad_event_ids)} "
    )
    if resolved.fingerprint:
        output += f"retained={len(result.retained_audio_ids)} songs={len(result.song_ids)} "
    output += f"issues={len(result.issues)}"
    typer.echo(output)


def ingest(
    source: SourceArgument,
    db_path: DbOption = None,
    fixture_transcript: FixtureTranscriptOption = None,
    source_url: SourceUrlOption = None,
    include_manifest_markers: MarkersOption = None,
    fingerprint: FingerprintOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Ingest a local source, optional deterministic transcript fixture, and optional markers into SQLite."""
    run_ingest_command(
        source,
        db_path=db_path,
        fixture_transcript=fixture_transcript,
        source_url=source_url,
        include_manifest_markers=include_manifest_markers,
        fingerprint=fingerprint,
        config_path=config_path,
    )


def _empty_counters() -> dict[str, int]:
    return {"segments": 0, "words": 0, "markers": 0, "issues": 0, "retained": 0, "songs": 0}


def _result_counters(result: IngestPipelineResult) -> dict[str, int]:
    return {
        "segments": len(result.segment_ids),
        "words": len(result.transcript_word_ids),
        "markers": len(result.ad_event_ids),
        "issues": len(result.issues),
        "retained": len(result.retained_audio_ids),
        "songs": len(result.song_ids),
    }


def _create_ingest_reporter(runtime_dir: Path, *, source: object) -> HealthReporter | None:
    try:
        return create_reporter(runtime_dir, command="ingest", source=source)
    except Exception:
        return None


def _ingest_progress_callback(reporter: HealthReporter | None):
    def record(progress: IngestPipelineProgress) -> None:
        if progress.phase == "completed":
            _report_finish(reporter, counters=progress.counters)
        elif progress.phase == "error":
            _report_fail(reporter, progress.error or "ingest failed", reason="pipeline_error", counters=progress.counters)
        else:
            _report_update(reporter, phase=progress.phase, counters=progress.counters)

    return record


def _report_start(reporter: HealthReporter | None) -> None:
    if reporter is None:
        return
    try:
        reporter.start(phase="setup", counters=_empty_counters())
    except Exception:
        pass


def _report_update(reporter: HealthReporter | None, *, phase: str, counters: dict[str, int]) -> None:
    if reporter is None:
        return
    try:
        reporter.update(phase=phase, counters=counters)
    except Exception:
        pass


def _report_finish(reporter: HealthReporter | None, *, counters: dict[str, int]) -> None:
    if reporter is None:
        return
    try:
        reporter.finish(phase="completed", reason="finished", counters=counters)
    except Exception:
        pass


def _report_fail(
    reporter: HealthReporter | None,
    error: object,
    *,
    reason: str,
    counters: dict[str, int] | None = None,
) -> None:
    if reporter is None:
        return
    try:
        reporter.fail(error, phase="error", reason=reason, counters=counters or _empty_counters())
    except Exception:
        pass


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = ["ingest", "run_ingest_command"]
