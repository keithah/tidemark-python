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
from tidemark.monitor_sources import monitor_source
from tidemark.runtime.health import HealthReporter, create_reporter, redact_source_label
from tidemark.runtime.logging import LifecycleLogger, resolve_lifecycle_log_path
from tidemark.runtime.retry import RetryPolicy


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
        logger = LifecycleLogger(resolve_lifecycle_log_path(config))
        _report_start(reporter)
        _log_lifecycle(logger, reporter, event="monitor.start", source=url, phase="setup", counters=_empty_counters())

        def marker_source_factory():
            return monitor_source(
                url,
                stream_type=resolved.stream_type,
                timeout=resolved.timeout_seconds,
            )

        result = run_monitor(
            marker_source_factory,
            options=RuntimeMonitorOptions(
                source_url=url,
                marker_filter=None if marker_filter is None else marker_filter.value,
                json_out=json_out,
                db_path=resolved_db_path,
                timeout=resolved.timeout_seconds,
                emit_summary=not quiet,
                progress_callback=_monitor_progress_callback(reporter, logger, source=url),
                retry_policy=RetryPolicy(
                    max_attempts=resolved.retry_attempts,
                    initial_backoff_seconds=resolved.retry_initial_backoff_seconds,
                    max_backoff_seconds=resolved.retry_max_backoff_seconds,
                ),
            ),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        _report_result(reporter, logger, result, source=url)
    except ConfigError as exc:
        _fatal(str(exc))
    except typer.Exit:
        raise
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


def _monitor_progress_callback(reporter: HealthReporter, logger: LifecycleLogger, *, source: str):
    retry_seen = False

    def record(progress: MonitorProgress) -> None:
        nonlocal retry_seen
        counters = _progress_counters(progress)
        if progress.phase == "running":
            _report_update(reporter, phase="running", counters=counters)
            if retry_seen:
                retry_seen = False
                _log_lifecycle(logger, reporter, event="monitor.reconnect", source=source, phase="running", counters=counters)
        elif progress.phase == "retrying":
            retry_seen = True
            _report_retry(
                reporter,
                attempt=getattr(progress, "retry_attempt", None) or 0,
                next_retry_at=getattr(progress, "next_retry_at", None),
                error=getattr(progress, "error", None) or "monitor retry",
                counters=counters,
            )
            _log_lifecycle(
                logger,
                reporter,
                event="monitor.retry",
                source=source,
                phase="retrying",
                counters=counters,
                retry_attempt=getattr(progress, "retry_attempt", None),
                next_retry_at=getattr(progress, "next_retry_at", None),
                error=getattr(progress, "error", None),
            )
        elif progress.phase == "error":
            return
        else:
            return

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


def _report_retry(
    reporter: HealthReporter,
    *,
    attempt: int,
    next_retry_at: object,
    error: object,
    counters: dict[str, int],
) -> None:
    try:
        reporter.retry(attempt=attempt, next_retry_at=next_retry_at, error=error, counters=counters)  # type: ignore[arg-type]
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


def _report_result(reporter: HealthReporter, logger: LifecycleLogger, result, *, source: str) -> None:
    counters = _result_counters(result)
    if result.reason == "error":
        _report_fail(reporter, result.error or "monitor error", phase="error", reason="error", counters=counters)
        _log_lifecycle(
            logger,
            reporter,
            event="monitor.terminal",
            source=source,
            phase="error",
            counters=counters,
            error=result.error or "monitor error",
            terminal_reason="error",
        )
    else:
        _report_finish(reporter, phase="completed", reason=result.reason, counters=counters)
        _log_lifecycle(
            logger,
            reporter,
            event="monitor.terminal",
            source=source,
            phase="completed",
            counters=counters,
            terminal_reason=result.reason,
        )


def _progress_counters(progress: MonitorProgress) -> dict[str, int]:
    counters = getattr(progress, "counters", None)
    if not isinstance(counters, dict):
        return _empty_counters()
    merged = _empty_counters()
    for key in merged:
        value = counters.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            merged[key] = value
    return merged


def _reporter_run_id(reporter: HealthReporter) -> str:
    record = getattr(reporter, "record", None)
    run_id = getattr(record, "run_id", None)
    return str(run_id) if run_id else "unknown"


def _log_lifecycle(
    logger: LifecycleLogger,
    reporter: HealthReporter,
    *,
    event: str,
    source: str,
    phase: str,
    counters: dict[str, int] | None = None,
    retry_attempt: int | None = None,
    next_retry_at: object | None = None,
    error: object | None = None,
    terminal_reason: object | None = None,
) -> None:
    try:
        logger.write(
            event=event,
            command="monitor",
            run_id=_reporter_run_id(reporter),
            source_label=source,
            phase=phase,
            counters=counters,
            retry_attempt=retry_attempt,
            next_retry_at=next_retry_at,  # type: ignore[arg-type]
            error=error,
            terminal_reason=terminal_reason,
        )
    except Exception:
        pass


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
