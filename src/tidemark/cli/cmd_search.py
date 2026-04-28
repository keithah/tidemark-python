"""Transcript search command implementation for the tidemark Typer CLI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from tidemark.search import TranscriptSearchError, search_transcript_db


QueryArgument = Annotated[
    str,
    typer.Argument(help="Transcript word or phrase to search for."),
]
DbOption = Annotated[
    Path,
    typer.Option("--db", help="SQLite database containing schema-v3 transcript_words rows."),
]
ContextOption = Annotated[
    float,
    typer.Option("--context", help="Seconds of surrounding transcript context to include."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit search results as one JSON array."),
]


def run_search_command(
    query: str,
    *,
    db_path: Path = Path("tidemark.db"),
    context_seconds: float = 5.0,
    json_output: bool = False,
) -> None:
    """Validate CLI-only inputs, delegate to the library, and format results."""
    if not query.strip():
        _fatal("query must be a non-empty string")
    if context_seconds < 0:
        _fatal("context_seconds must be a non-negative number")

    try:
        results = search_transcript_db(db_path, query, context_seconds=context_seconds)
    except TranscriptSearchError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("search failed")

    if json_output:
        rows = [asdict(result) for result in results]
        typer.echo(json.dumps(rows, separators=(",", ":")))
        return

    if not results:
        typer.echo("No transcript matches found.")
        return

    for result in results:
        typer.echo(
            f"{result.source_url} | {result.hit_start_ts:.3f}s | "
            f"segment {result.segment_id} seq {result.segment_sequence} | {result.context_text}"
        )


def search(
    query: QueryArgument,
    db_path: DbOption = Path("tidemark.db"),
    context_seconds: ContextOption = 5.0,
    json_output: JsonOption = False,
) -> None:
    """Search stored transcript words and print timestamped context windows."""
    run_search_command(
        query,
        db_path=db_path,
        context_seconds=context_seconds,
        json_output=json_output,
    )


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = ["run_search_command", "search"]
