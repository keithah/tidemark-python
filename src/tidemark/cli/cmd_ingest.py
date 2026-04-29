"""Deterministic ingest command implementation for the tidemark Typer CLI."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from tidemark.config import ConfigError, IngestOverrides, default_runtime_dir, load_config, resolve_ingest_options
from tidemark.ingest import SegmentIngestError
from tidemark.ingest.pipeline import (
    IngestPipelineProgress,
    IngestPipelineResult,
    TranscriptFixtureError,
    ingest_live_hls_to_db,
    ingest_source_to_db,
    load_fixture_transcript,
)
from tidemark.runtime.health import HealthReporter, create_reporter
from tidemark.transcribe import AppleSpeechTranscriber, AppleSpeechUnavailable, DeterministicTranscriber


class CliTranscriber(str, Enum):
    """User-facing live transcription backend choices."""

    APPLE = "apple"


SourceArgument = Annotated[
    str,
    typer.Argument(help="Local manifest/media source or live HLS URL to ingest."),
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
TranscribeOption = Annotated[
    bool | None,
    typer.Option("--transcribe/--no-transcribe", help="Enable or disable live speech transcription for network HLS ingest."),
]
TranscriberOption = Annotated[
    CliTranscriber,
    typer.Option("--transcriber", help="Speech transcription backend for --transcribe."),
]
TimeoutOption = Annotated[
    float | None,
    typer.Option("--timeout", min=0.0, help="Stop live network ingest after this many seconds."),
]
VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Print live ingest progress and debug tracebacks to stderr."),
]


def run_ingest_command(
    source: str,
    *,
    db_path: Path | None = None,
    fixture_transcript: Path | None = None,
    source_url: str | None = None,
    include_manifest_markers: bool | None = None,
    fingerprint: bool | None = None,
    config_path: Path | None = None,
    transcribe: bool | None = None,
    transcriber_backend: CliTranscriber = CliTranscriber.APPLE,
    timeout: float | None = None,
    verbose: bool = False,
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
    effective_transcribe = transcribe if transcribe is not None else _is_network_hls_source(source)
    if fixture_transcript is not None:
        try:
            fixture_words = load_fixture_transcript(fixture_transcript)
        except TranscriptFixtureError as exc:
            _report_fail(reporter, str(exc), reason="fixture_error")
            _fatal(str(exc))
        except Exception:
            _report_fail(reporter, "fixture transcript could not be loaded", reason="fixture_error")
            _fatal("fixture transcript could not be loaded")
    elif not resolved.fingerprint and not effective_transcribe:
        _report_fail(
            reporter,
            "--fixture-transcript is required unless --fingerprint or --transcribe is enabled",
            reason="fixture_error",
        )
        _fatal("--fixture-transcript is required unless --fingerprint or --transcribe is enabled")

    transcriber = None
    if effective_transcribe:
        try:
            if transcriber_backend is CliTranscriber.APPLE:
                transcriber = AppleSpeechTranscriber()
            else:  # pragma: no cover - enum guards this path.
                _fatal("unsupported transcriber backend")
        except AppleSpeechUnavailable as exc:
            _report_fail(reporter, str(exc), reason="transcriber_error")
            if verbose:
                import traceback

                traceback.print_exc()
            _fatal(str(exc))
    elif fixture_words is not None:
        transcriber = DeterministicTranscriber(
            fixture_words,
            language="en",
            engine="deterministic-fixture",
        )

    try:
        progress_callback = _ingest_progress_callback(reporter, verbose=verbose)
        if effective_transcribe:
            if transcriber is None:
                _fatal("transcriber setup failed")
            result = ingest_live_hls_to_db(
                source,
                db_path=resolved.db_path,
                transcriber=transcriber,
                timeout=timeout,
                progress_callback=progress_callback,
            )
        else:
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
                progress_callback=progress_callback,
            )
    except SegmentIngestError as exc:
        _report_fail(reporter, str(exc), reason="source_error")
        if verbose:
            import traceback

            traceback.print_exc()
        _fatal(str(exc))
    except Exception as exc:
        _report_fail(reporter, str(exc), reason="pipeline_error")
        if verbose:
            import traceback

            traceback.print_exc()
        _fatal("ingest failed")

    final_counters = _result_counters(result)
    _report_finish(reporter, counters=final_counters)
    counters = _result_counters(result)
    output = (
        "Ingest complete: "
        f"segments={counters['segments']} "
        f"processed={counters['processed']} "
        f"skipped={counters['skipped']} "
        f"failed={counters['failed']} "
        f"words={counters['words']} "
        f"markers={counters['markers']} "
    )
    if resolved.fingerprint:
        output += f"retained={counters['retained']} songs={counters['songs']} "
    output += f"issues={counters['issues']}"
    typer.echo(output)


def ingest(
    source: SourceArgument,
    db_path: DbOption = None,
    fixture_transcript: FixtureTranscriptOption = None,
    source_url: SourceUrlOption = None,
    include_manifest_markers: MarkersOption = None,
    fingerprint: FingerprintOption = None,
    config_path: ConfigOption = None,
    transcribe: TranscribeOption = None,
    transcriber_backend: TranscriberOption = CliTranscriber.APPLE,
    timeout: TimeoutOption = None,
    verbose: VerboseOption = False,
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
        transcribe=transcribe,
        transcriber_backend=transcriber_backend,
        timeout=timeout,
        verbose=verbose,
    )


def _is_network_hls_source(source: str) -> bool:
    lowered = str(source).lower().split("?", 1)[0]
    return lowered.startswith(("http://", "https://")) and lowered.endswith(".m3u8")


def _empty_counters() -> dict[str, int]:
    return {
        "segments": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "words": 0,
        "markers": 0,
        "issues": 0,
        "retained": 0,
        "songs": 0,
    }


def _result_counters(result: IngestPipelineResult) -> dict[str, int]:
    processed = _field_count(result, "segment_ids")
    skipped = _field_count(result, "skipped_segment_ids")
    failed = _field_count(result, "issues")
    return {
        "segments": processed + skipped,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "words": _field_count(result, "transcript_word_ids"),
        "markers": _field_count(result, "ad_event_ids"),
        "issues": failed,
        "retained": _field_count(result, "retained_audio_ids"),
        "songs": _field_count(result, "song_ids"),
    }


def _field_count(result: object, field_name: str) -> int:
    value = getattr(result, field_name, ())
    return len(value) if value is not None else 0


def _create_ingest_reporter(runtime_dir: Path, *, source: object) -> HealthReporter | None:
    try:
        return create_reporter(runtime_dir, command="ingest", source=source)
    except Exception:
        return None


def _ingest_progress_callback(reporter: HealthReporter | None, *, verbose: bool = False):
    def record(progress: IngestPipelineProgress) -> None:
        if verbose:
            counters = ",".join(f"{key}={value}" for key, value in sorted(progress.counters.items()))
            typer.echo(f"[tidemark] ingest phase={progress.phase} {counters}", err=True)
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
