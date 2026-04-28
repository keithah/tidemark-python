"""Monitor command implementation for the tidemark Typer CLI."""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from tidemark.monitor import MonitorOptions, run_monitor
from tidemark.monitor_sources import MonitorSourceError, monitor_source


class CliStreamType(str, Enum):
    """User-facing stream type choices."""

    AUTO = "auto"
    HLS = "hls"
    ICECAST = "icecast"
    ICY = "icy"
    MPEGTS = "mpegts"
    UDP = "udp"


class CliMarkerFilter(str, Enum):
    """User-facing marker type filters."""

    SCTE35 = "scte35"
    ID3 = "id3"
    ICY = "icy"


UrlArgument = Annotated[
    str,
    typer.Argument(help="Stream URL, UDP address, MPEG-TS file, or HLS manifest to monitor."),
]
StreamTypeOption = Annotated[
    CliStreamType,
    typer.Option("--stream-type", help="Force source type instead of auto-detection."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit marker records as NDJSON on stdout. Accepted for Go CLI compatibility."),
]
QuietOption = Annotated[
    bool,
    typer.Option("--quiet", help="Suppress stderr completion summaries. Marker NDJSON is still emitted."),
]
FilterOption = Annotated[
    CliMarkerFilter | None,
    typer.Option("--filter", help="Emit only the selected marker type."),
]
JsonOutOption = Annotated[
    Path | None,
    typer.Option("--json-out", help="Mirror emitted marker NDJSON to this file."),
]
TimeoutOption = Annotated[
    float | None,
    typer.Option("--timeout", min=0.0, help="Stop monitoring after this many seconds."),
]
DbOption = Annotated[
    Path | None,
    typer.Option("--db", help="Persist emitted markers to a SQLite database."),
]


def run_monitor_command(
    url: str,
    *,
    stream_type: CliStreamType = CliStreamType.AUTO,
    json_output: bool = False,
    quiet: bool = False,
    marker_filter: CliMarkerFilter | None = None,
    json_out: Path | None = None,
    timeout: float | None = None,
    db_path: Path | None = None,
) -> None:
    """Convert CLI options and delegate to importable monitor/source layers."""
    _ = json_output  # Kept for compatibility; monitor output is always NDJSON.
    try:
        marker_source = monitor_source(
            url,
            stream_type=stream_type.value,
            timeout=timeout,
        )
        result = run_monitor(
            marker_source,
            options=MonitorOptions(
                source_url=url,
                marker_filter=None if marker_filter is None else marker_filter.value,
                json_out=json_out,
                db_path=db_path,
                timeout=timeout,
                emit_summary=not quiet,
            ),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except MonitorSourceError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("monitor failed")

    if result.reason == "error":
        raise typer.Exit(code=1)


def monitor(
    url: UrlArgument,
    stream_type: StreamTypeOption = CliStreamType.AUTO,
    json_output: JsonOption = False,
    quiet: QuietOption = False,
    marker_filter: FilterOption = None,
    json_out: JsonOutOption = None,
    timeout: TimeoutOption = None,
    db_path: DbOption = None,
) -> None:
    """Monitor a stream or local source for ad markers."""
    run_monitor_command(
        url,
        stream_type=stream_type,
        json_output=json_output,
        quiet=quiet,
        marker_filter=marker_filter,
        json_out=json_out,
        timeout=timeout,
        db_path=db_path,
    )


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)
