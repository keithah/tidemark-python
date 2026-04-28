"""Retained-audio clip export command implementation for the tidemark Typer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tidemark.clip import ClipExportError, export_clip_db
from tidemark.config import ClipOverrides, ConfigError, load_config, resolve_clip_options


AtOption = Annotated[
    float,
    typer.Option("--at", help="Timestamp in seconds to center the exported clip around."),
]
ContextOption = Annotated[
    float,
    typer.Option("--context", help="Seconds of retained audio context to include before and after --at."),
]
DbOption = Annotated[
    Path | None,
    typer.Option("--db", help="SQLite database containing schema-v4 retained_audio rows."),
]
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="TOML config file to load for command defaults."),
]
OutOption = Annotated[
    Path,
    typer.Option("--out", help="Destination WAV path for the exported clip."),
]


def run_clip_command(
    *,
    at_seconds: float,
    context_seconds: float,
    db_path: Path = Path("tidemark.db"),
    out_path: Path,
) -> None:
    """Validate CLI-only inputs, delegate to the library, and format metadata."""
    if at_seconds < 0:
        _fatal("at_seconds must be a non-negative number")
    if context_seconds < 0:
        _fatal("context_seconds must be a non-negative number")

    try:
        result = export_clip_db(
            db_path,
            at_seconds=at_seconds,
            context_seconds=context_seconds,
            out_path=out_path,
        )
    except ClipExportError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("clip export failed")

    typer.echo(
        "Clip exported: "
        f"start={result.start_ts:.3f} "
        f"duration={result.duration_seconds:.3f} "
        f"bytes={result.byte_length} "
        f"sha256={result.sha256}"
    )


def clip(
    at_seconds: AtOption,
    context_seconds: ContextOption,
    out_path: OutOption,
    db_path: DbOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Export a WAV clip from retained schema-v4 audio around a timestamp."""
    try:
        config = load_config(config_path, explicit=config_path is not None)
        resolved = resolve_clip_options(config, ClipOverrides(db_path=db_path))
    except ConfigError as exc:
        _fatal(str(exc))

    run_clip_command(
        at_seconds=at_seconds,
        context_seconds=context_seconds,
        db_path=resolved.db_path,
        out_path=out_path,
    )


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = ["clip", "run_clip_command"]
