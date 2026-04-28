"""Report command group implementation for the tidemark Typer CLI."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer

from tidemark.reports import (
    AdSummaryReportRow,
    PlayReportRow,
    RepeatReportRow,
    ReportError,
    ads_report_db,
    plays_report_db,
    repeats_report_db,
)


DbOption = Annotated[
    Path,
    typer.Option("--db", help="SQLite database containing schema-v4 timeline rows."),
]
SinceOption = Annotated[
    float | None,
    typer.Option("--since", help="Only include rows at or after this stream-relative timestamp in seconds."),
]
SourceOption = Annotated[
    str | None,
    typer.Option("--source", help="Only include rows for this source URL."),
]
MinScoreOption = Annotated[
    float,
    typer.Option("--min-score", help="Minimum song identification score, inclusive, from 0 to 1."),
]
MinCountOption = Annotated[
    int,
    typer.Option("--min-count", help="Minimum repeated play count, inclusive."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit report rows as one compact JSON array."),
]

report = typer.Typer(
    add_completion=False,
    help="Report identified plays, repeat airings, and ad summaries from SQLite timeline rows.",
    invoke_without_command=True,
)


@report.callback()
def report_root(ctx: typer.Context) -> None:
    """Report identified plays, repeat airings, and ad summaries from SQLite timeline rows."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


def run_plays_report_command(
    *,
    db_path: Path = Path("tidemark.db"),
    since_seconds: float | None = None,
    source_url: str | None = None,
    min_score: float = 0.8,
    json_output: bool = False,
) -> None:
    """Validate CLI-only inputs, delegate to play report APIs, and format results."""
    _validate_since(since_seconds)
    _validate_min_score(min_score)
    try:
        rows = plays_report_db(
            db_path,
            since_seconds=since_seconds,
            source_url=source_url,
            min_score=min_score,
        )
    except ReportError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("report failed")

    _emit_json_or_human(
        rows,
        json_output=json_output,
        empty_message="No identified plays found.",
        format_row=_format_play_row,
    )


def run_repeats_report_command(
    *,
    db_path: Path = Path("tidemark.db"),
    since_seconds: float | None = None,
    source_url: str | None = None,
    min_count: int = 2,
    min_score: float = 0.8,
    json_output: bool = False,
) -> None:
    """Validate CLI-only inputs, delegate to repeat report APIs, and format results."""
    _validate_since(since_seconds)
    _validate_min_count(min_count)
    _validate_min_score(min_score)
    try:
        rows = repeats_report_db(
            db_path,
            since_seconds=since_seconds,
            source_url=source_url,
            min_count=min_count,
            min_score=min_score,
        )
    except ReportError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("report failed")

    _emit_json_or_human(
        rows,
        json_output=json_output,
        empty_message="No repeat plays found.",
        format_row=_format_repeat_row,
    )


def run_ads_report_command(
    *,
    db_path: Path = Path("tidemark.db"),
    since_seconds: float | None = None,
    source_url: str | None = None,
    json_output: bool = False,
) -> None:
    """Validate CLI-only inputs, delegate to ad report APIs, and format results."""
    _validate_since(since_seconds)
    try:
        rows = ads_report_db(db_path, since_seconds=since_seconds, source_url=source_url)
    except ReportError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("report failed")

    _emit_json_or_human(
        rows,
        json_output=json_output,
        empty_message="No ad summaries found.",
        format_row=_format_ad_row,
    )


@report.command(name="plays")
def plays(
    db_path: DbOption = Path("tidemark.db"),
    since_seconds: SinceOption = None,
    source_url: SourceOption = None,
    min_score: MinScoreOption = 0.8,
    json_output: JsonOption = False,
) -> None:
    """Print identified song plays from the persisted timeline."""
    run_plays_report_command(
        db_path=db_path,
        since_seconds=since_seconds,
        source_url=source_url,
        min_score=min_score,
        json_output=json_output,
    )


@report.command(name="repeats")
def repeats(
    db_path: DbOption = Path("tidemark.db"),
    since_seconds: SinceOption = None,
    source_url: SourceOption = None,
    min_count: MinCountOption = 2,
    min_score: MinScoreOption = 0.8,
    json_output: JsonOption = False,
) -> None:
    """Print repeated identified song groups from the persisted timeline."""
    run_repeats_report_command(
        db_path=db_path,
        since_seconds=since_seconds,
        source_url=source_url,
        min_count=min_count,
        min_score=min_score,
        json_output=json_output,
    )


@report.command(name="ads")
def ads(
    db_path: DbOption = Path("tidemark.db"),
    since_seconds: SinceOption = None,
    source_url: SourceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Print grouped ad marker summaries from the persisted timeline."""
    run_ads_report_command(
        db_path=db_path,
        since_seconds=since_seconds,
        source_url=source_url,
        json_output=json_output,
    )


def _emit_json_or_human[RowT](
    rows: tuple[RowT, ...],
    *,
    json_output: bool,
    empty_message: str,
    format_row,
) -> None:
    if json_output:
        typer.echo(json.dumps([_public_report_dict(row) for row in rows], separators=(",", ":")))
        return

    if not rows:
        typer.echo(empty_message)
        return

    for row in rows:
        typer.echo(format_row(row))


def _public_report_dict(row: object) -> dict[str, object]:
    values = asdict(row)
    if "source_url" in values:
        values["source_url"] = _public_source_label(str(values["source_url"]))
    if "source_urls" in values:
        values["source_urls"] = [_public_source_label(str(source_url)) for source_url in values["source_urls"]]
    return values


def _public_source_label(source_url: str) -> str:
    parsed = urlparse(source_url)
    if parsed.scheme == "file":
        return "[local file]"
    return source_url


def _format_play_row(row: PlayReportRow) -> str:
    end_ts = row.start_ts + row.duration_seconds
    parts = [
        _public_source_label(row.source_url),
        f"{row.start_ts:.3f}s-{end_ts:.3f}s",
        f"segment {row.segment_id} seq {row.segment_sequence}",
        row.title,
    ]
    if row.artist:
        parts.append(f"artist {row.artist}")
    parts.append(f"score {row.score:.3f}")
    return " | ".join(parts)


def _format_repeat_row(row: RepeatReportRow) -> str:
    parts = [row.title]
    if row.artist:
        parts.append(f"artist {row.artist}")
    parts.extend(
        [
            f"identity {row.identity}",
            f"count {row.count}",
            f"first {row.first_start_ts:.3f}s",
            f"last {row.last_start_ts:.3f}s",
            f"sources {', '.join(sorted(_public_source_label(source_url) for source_url in row.source_urls))}",
            f"best_score {row.best_score:.3f}",
        ]
    )
    return " | ".join(parts)


def _format_ad_row(row: AdSummaryReportRow) -> str:
    return " | ".join(
        [
            _public_source_label(row.source_url),
            row.classification,
            row.marker_type,
            f"count {row.count}",
            f"first {row.first_ts:.3f}s",
            f"last {row.last_ts:.3f}s",
            f"total_break_duration {row.total_break_duration:.3f}s",
        ]
    )


def _validate_since(since_seconds: float | None) -> None:
    if since_seconds is not None and since_seconds < 0:
        _fatal("since_seconds must be a non-negative number")


def _validate_min_score(min_score: float) -> None:
    if min_score < 0 or min_score > 1:
        _fatal("min_score must be between 0 and 1")


def _validate_min_count(min_count: int) -> None:
    if min_count < 1:
        _fatal("min_count must be >= 1")


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = [
    "ads",
    "plays",
    "repeats",
    "report",
    "run_ads_report_command",
    "run_plays_report_command",
    "run_repeats_report_command",
]
