"""Deterministic ingest command implementation for the tidemark Typer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tidemark.config import ConfigError, IngestOverrides, load_config, resolve_ingest_options
from tidemark.ingest import SegmentIngestError
from tidemark.ingest.pipeline import TranscriptFixtureError, ingest_source_to_db, load_fixture_transcript
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

    fixture_words = None
    if fixture_transcript is not None:
        try:
            fixture_words = load_fixture_transcript(fixture_transcript)
        except TranscriptFixtureError as exc:
            _fatal(str(exc))
        except Exception:
            _fatal("fixture transcript could not be loaded")
    elif not resolved.fingerprint:
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
        )
    except SegmentIngestError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("ingest failed")

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


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = ["ingest", "run_ingest_command"]
