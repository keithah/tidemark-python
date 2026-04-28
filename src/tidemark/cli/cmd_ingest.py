"""Deterministic ingest command implementation for the tidemark Typer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tidemark.ingest import SegmentIngestError
from tidemark.ingest.pipeline import TranscriptFixtureError, ingest_source_to_db, load_fixture_transcript
from tidemark.transcribe import DeterministicTranscriber


SourceArgument = Annotated[
    Path,
    typer.Argument(help="Local manifest or media source to ingest."),
]
DbOption = Annotated[
    Path,
    typer.Option("--db", help="SQLite database to create or update."),
]
FixtureTranscriptOption = Annotated[
    Path,
    typer.Option("--fixture-transcript", help="Deterministic transcript JSON fixture for M002 ingest proof."),
]
SourceUrlOption = Annotated[
    str | None,
    typer.Option("--source-url", help="Optional logical source URL stored with segment/transcript rows."),
]
NoMarkersOption = Annotated[
    bool,
    typer.Option("--no-markers", help="Disable manifest marker persistence."),
]


def run_ingest_command(
    source: Path,
    *,
    db_path: Path = Path("tidemark.db"),
    fixture_transcript: Path,
    source_url: str | None = None,
    include_manifest_markers: bool = True,
) -> None:
    """Load deterministic fixtures, delegate once to the pipeline, and print safe counts."""
    try:
        fixture_words = load_fixture_transcript(fixture_transcript)
    except TranscriptFixtureError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("fixture transcript could not be loaded")

    transcriber = DeterministicTranscriber(
        fixture_words,
        language="en",
        engine="deterministic-fixture",
    )

    try:
        result = ingest_source_to_db(
            source,
            db_path=db_path,
            transcriber=transcriber,
            source_url=source_url,
            include_manifest_markers=include_manifest_markers,
        )
    except SegmentIngestError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("ingest failed")

    typer.echo(
        "Ingest complete: "
        f"segments={len(result.segment_ids)} "
        f"words={len(result.transcript_word_ids)} "
        f"markers={len(result.ad_event_ids)} "
        f"issues={len(result.issues)}"
    )


def ingest(
    source: SourceArgument,
    db_path: DbOption = Path("tidemark.db"),
    fixture_transcript: FixtureTranscriptOption = ...,  # type: ignore[assignment]
    source_url: SourceUrlOption = None,
    no_markers: NoMarkersOption = False,
) -> None:
    """Ingest a local source, deterministic transcript fixture, and optional markers into SQLite."""
    run_ingest_command(
        source,
        db_path=db_path,
        fixture_transcript=fixture_transcript,
        source_url=source_url,
        include_manifest_markers=not no_markers,
    )


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = ["ingest", "run_ingest_command"]
