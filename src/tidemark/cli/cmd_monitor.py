"""Monitor command implementation for the tidemark Typer CLI."""

from __future__ import annotations

import os
import sys
import tomllib
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from tidemark.config import ConfigError, MonitorOverrides, default_runtime_dir, load_config, resolve_monitor_options
from tidemark.monitor import MonitorOptions as RuntimeMonitorOptions
from tidemark.monitor import MonitorProgress
from tidemark.monitor import run_monitor
from tidemark.monitor_sources import MonitorSourceError, monitor_source
from tidemark.runtime.health import HealthReporter, create_reporter, redact_source_label


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
    CliStreamType | None,
    typer.Option("--stream-type", help="Force source type instead of auto-detection."),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit marker records as NDJSON on stdout. Accepted for Go CLI compatibility."),
]
QuietOption = Annotated[
    bool,
    typer.Option("--quiet/--no-quiet", help="Suppress stderr completion summaries. Marker NDJSON is still emitted."),
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
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="TOML config file to load for command defaults."),
]


def run_monitor_command(
    url: str,
    *,
    stream_type: CliStreamType | None = None,
    json_output: bool = False,
    quiet: bool = False,
    marker_filter: CliMarkerFilter | None = None,
    json_out: Path | None = None,
    timeout: float | None = None,
    db_path: Path | None = None,
    config_path: Path | None = None,
) -> None:
    """Convert CLI options and delegate to importable monitor/source layers."""
    _ = json_output  # Kept for compatibility; monitor output is always NDJSON.
    try:
        config = load_config(config_path, explicit=config_path is not None)
        resolved = resolve_monitor_options(
            config,
            MonitorOverrides(
                db_path=db_path,
                stream_type=None if stream_type is None else stream_type.value,
                timeout_seconds=timeout,
            ),
        )
        resolved_db_path: Path | None = resolved.db_path
        if db_path is None and "TIDEMARK_DB" not in os.environ and not _config_declares_paths_db(config_path):
            resolved_db_path = None

        runtime_dir = Path(config.paths.runtime_dir or default_runtime_dir()).expanduser()
        reporter = create_reporter(runtime_dir, command="monitor", source=url)
        _report_start(reporter)

        try:
            marker_source = monitor_source(
                url,
                stream_type=resolved.stream_type,
                timeout=resolved.timeout_seconds,
            )
        except MonitorSourceError as exc:
            _report_fail(reporter, str(exc), phase="error", reason="source_setup_error", counters=_empty_counters())
            _fatal(redact_source_label(str(exc)))

        result = run_monitor(
            marker_source,
            options=RuntimeMonitorOptions(
                source_url=url,
                marker_filter=None if marker_filter is None else marker_filter.value,
                json_out=json_out,
                db_path=resolved_db_path,
                timeout=resolved.timeout_seconds,
                emit_summary=not quiet,
                progress_callback=_monitor_progress_callback(reporter),
            ),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        _report_result(reporter, result)
    except ConfigError as exc:
        _fatal(str(exc))
    except Exception:
        _fatal("monitor failed")

    if result.reason == "error":
        raise typer.Exit(code=1)


def monitor(
    url: UrlArgument,
    stream_type: StreamTypeOption = None,
    json_output: JsonOption = False,
    quiet: QuietOption = False,
    marker_filter: FilterOption = None,
    json_out: JsonOutOption = None,
    timeout: TimeoutOption = None,
    db_path: DbOption = None,
    config_path: ConfigOption = None,
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
        config_path=config_path,
    )


def _empty_counters() -> dict[str, int]:
    return {"markers_seen": 0, "markers_emitted": 0, "markers_filtered": 0, "sink_warnings": 0}


def _monitor_progress_callback(reporter: HealthReporter):
    def record(progress: MonitorProgress) -> None:
        if progress.phase == "running":
            _report_update(reporter, phase="running", counters=progress.counters)
        elif progress.phase == "error":
            _report_fail(
                reporter,
                progress.error or progress.reason or "monitor error",
                phase="error",
                reason=progress.reason or "error",
                counters=progress.counters,
            )
        else:
            _report_finish(
                reporter,
                phase="completed",
                reason=progress.reason or "finished",
                counters=progress.counters,
            )

    return record


def _report_start(reporter: HealthReporter) -> None:
    try:
        reporter.start(phase="setup", counters=_empty_counters())
    except Exception:
        pass


def _report_update(reporter: HealthReporter, *, phase: str, counters: dict[str, int]) -> None:
    try:
        reporter.update(phase=phase, counters=counters)
    except Exception:
        pass


def _report_finish(reporter: HealthReporter, *, phase: str, reason: str, counters: dict[str, int]) -> None:
    try:
        reporter.finish(phase=phase, reason=reason, counters=counters)
    except Exception:
        pass


def _report_fail(reporter: HealthReporter, error: object, *, phase: str, reason: str, counters: dict[str, int]) -> None:
    try:
        reporter.fail(error, phase=phase, reason=reason, counters=counters)
    except Exception:
        pass


def _result_counters(result) -> dict[str, int]:
    return {
        "markers_seen": result.markers_seen,
        "markers_emitted": result.markers_emitted,
        "markers_filtered": result.markers_filtered,
        "sink_warnings": result.sink_warnings,
    }


def _report_result(reporter: HealthReporter, result) -> None:
    counters = _result_counters(result)
    if result.reason == "error":
        _report_fail(reporter, result.error or "monitor error", phase="error", reason="error", counters=counters)
    else:
        _report_finish(reporter, phase="completed", reason=result.reason, counters=counters)


def _config_declares_paths_db(config_path: Path | None) -> bool:
    selected = config_path
    if selected is None:
        env_path = os.environ.get("TIDEMARK_CONFIG")
        if not env_path:
            return False
        selected = Path(env_path)
    try:
        with Path(selected).expanduser().open("rb") as handle:
            raw = tomllib.load(handle)
    except Exception:
        return False
    paths = raw.get("paths") if isinstance(raw, dict) else None
    return isinstance(paths, dict) and "db" in paths


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)
