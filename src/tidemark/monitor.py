"""Importable monitor runtime orchestration.

This module intentionally contains no CLI parsing, terminal styling, network clients, or
source adapters. Callers provide an iterable (or zero-argument callable returning an
iterable) of :class:`tidemark.markers.AdMarker` values plus injected output streams.
"""

from __future__ import annotations

import re
import time
import traceback as _traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Literal, TextIO
from urllib.parse import urlparse

from tidemark.markers import AD_END, AD_START, UNKNOWN, AdMarker, Classifier
from tidemark.monitor_sources import MonitorSourceError, StreamType
from tidemark.runtime.health import redact_source_label
from tidemark.runtime.retry import RetryPolicy
import tidemark.store.db as db

MonitorReason = Literal["eof", "timeout", "interrupted", "error"]
MarkerFilter = Literal["all", "ad", "AD_START", "AD_END", "UNKNOWN", "scte35", "id3", "icy"]
MonitorPhase = Literal["running", "retrying", "completed", "error"]
RetrySleep = Callable[[float], None]


@dataclass(frozen=True)
class MonitorProgress:
    """Best-effort progress snapshot for runtime health observers."""

    phase: MonitorPhase
    counters: dict[str, int]
    reason: MonitorReason | None = None
    error: str | None = None
    retry_attempt: int | None = None
    delay_seconds: float | None = None
    next_retry_at: datetime | None = None


MonitorProgressCallback = Callable[[MonitorProgress], None]

_MARKER_TYPE_FILTERS = {"scte35", "id3", "icy"}
_VALID_FILTERS = {"all", "ad", AD_START, AD_END, UNKNOWN, *_MARKER_TYPE_FILTERS}
_RETRYABLE_STREAM_TYPES = {StreamType.HLS, StreamType.ICY, StreamType.UDP}
_PAYLOAD_ASSIGNMENT_RE = re.compile(r"(?i)\b(raw[_-]?base64|payload)\s*[=:]\s*[^\s&]+")
_EMBEDDED_PATH_RE = re.compile(r"(?<![A-Za-z0-9+.-])(?:~|/|\.\.?/)[^\s]+")


@dataclass(frozen=True)
class MonitorOptions:
    """Options for the importable monitor runtime."""

    source_url: str = ""
    marker_filter: str | None = "all"
    json_out: str | PathLike[str] | None = None
    db_path: str | PathLike[str] | None = None
    timeout: float | None = None
    clock: Callable[[], float] = time.monotonic
    emit_summary: bool = True
    verbose: bool = False
    progress_callback: MonitorProgressCallback | None = None
    retry_policy: RetryPolicy | None = None
    retry_sleep: RetrySleep = time.sleep


@dataclass(frozen=True)
class MonitorResult:
    """Result counters and terminal reason for one monitor run."""

    reason: MonitorReason
    markers_seen: int
    markers_emitted: int
    markers_filtered: int
    sink_warnings: int = 0
    error: str | None = None


@dataclass
class _MonitorState:
    markers_seen: int = 0
    markers_emitted: int = 0
    markers_filtered: int = 0
    sink_warnings: int = 0


@dataclass(frozen=True)
class _RetryOutcome:
    retried: bool
    terminal_reason: MonitorReason | None = None
    terminal_error: str | None = None


def run_monitor(
    marker_source: Iterable[AdMarker] | Callable[[], Iterable[AdMarker]],
    *,
    options: MonitorOptions | None = None,
    stdout: TextIO,
    stderr: TextIO,
) -> MonitorResult:
    """Classify, filter, emit, and optionally persist ad markers.

    ``marker_source`` is deliberately already-opened/injected so this runtime can be
    tested and reused without importing source adapters or parsing CLI arguments.
    """
    active_options = options or MonitorOptions()
    state = _MonitorState()

    normalized_filter = _normalize_filter(active_options.marker_filter)
    if normalized_filter is None:
        return _fatal_result(
            "invalid marker filter",
            state,
            stderr,
            emit_summary=active_options.emit_summary,
            progress_callback=active_options.progress_callback,
        )

    json_handle: TextIO | None = None
    db_conn: object | None = None
    try:
        json_handle = _open_json_out(active_options.json_out)
    except Exception:
        return _fatal_result(
            "json-out setup failed",
            state,
            stderr,
            emit_summary=active_options.emit_summary,
            progress_callback=active_options.progress_callback,
        )

    try:
        db_conn = _open_db(active_options.db_path)
    except Exception:
        _close_handle(json_handle)
        return _fatal_result(
            "database setup failed",
            state,
            stderr,
            emit_summary=active_options.emit_summary,
            progress_callback=active_options.progress_callback,
        )

    classifier = Classifier()
    start_time = active_options.clock()
    next_retry_attempt = 1
    _notify_progress(active_options.progress_callback, "running", state)

    try:
        while True:
            try:
                iterator = _iter_marker_source(marker_source)
            except KeyboardInterrupt:
                return _finish(
                    "interrupted",
                    state,
                    stderr,
                    emit_summary=active_options.emit_summary,
                    progress_callback=active_options.progress_callback,
                )
            except Exception as exc:
                if active_options.verbose:
                    _traceback.print_exc(file=stderr)
                retry = _maybe_retry_source_failure(
                    exc,
                    next_retry_attempt,
                    state,
                    active_options,
                    start_time,
                )
                if retry.retried:
                    next_retry_attempt += 1
                    continue
                if retry.terminal_reason == "timeout":
                    return _finish(
                        "timeout",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )
                return _fatal_result(
                    retry.terminal_error or _source_error_message(exc, fallback="marker source setup failed"),
                    state,
                    stderr,
                    emit_summary=active_options.emit_summary,
                    progress_callback=active_options.progress_callback,
                )

            while True:
                if _timed_out(active_options, start_time):
                    return _finish(
                        "timeout",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )

                try:
                    item = next(iterator)
                except StopIteration:
                    return _finish(
                        "eof",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )
                except KeyboardInterrupt:
                    return _finish(
                        "interrupted",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )
                except Exception as exc:
                    if active_options.verbose:
                        _traceback.print_exc(file=stderr)
                    retry = _maybe_retry_source_failure(
                        exc,
                        next_retry_attempt,
                        state,
                        active_options,
                        start_time,
                    )
                    if retry.retried:
                        next_retry_attempt += 1
                        break
                    if retry.terminal_reason == "timeout":
                        return _finish(
                            "timeout",
                            state,
                            stderr,
                            emit_summary=active_options.emit_summary,
                            progress_callback=active_options.progress_callback,
                        )
                    return _fatal_result(
                        retry.terminal_error or _source_error_message(exc, fallback="marker iterator failed"),
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )

                if not isinstance(item, AdMarker):
                    return _fatal_result(
                        "marker source yielded non-AdMarker value",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )

                state.markers_seen += 1
                classifier.classify(item)
                if not _matches_filter(item, normalized_filter):
                    state.markers_filtered += 1
                    _notify_progress(active_options.progress_callback, "running", state)
                    continue

                try:
                    raw_json = item.to_json()
                except Exception:
                    return _fatal_result(
                        "marker serialization failed",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )

                try:
                    stdout.write(raw_json)
                    stdout.write("\n")
                    stdout.flush()
                except Exception:
                    return _fatal_result(
                        "stdout write failed",
                        state,
                        stderr,
                        emit_summary=active_options.emit_summary,
                        progress_callback=active_options.progress_callback,
                    )

                state.markers_emitted += 1
                if json_handle is not None:
                    try:
                        json_handle.write(raw_json)
                        json_handle.write("\n")
                        json_handle.flush()
                    except Exception:
                        state.sink_warnings += 1
                        _warn("json-out write failed", stderr)

                if db_conn is not None:
                    try:
                        db.insert_ad_event(db_conn, active_options.source_url, item)  # type: ignore[arg-type]
                    except Exception:
                        state.sink_warnings += 1
                        _warn("database write failed", stderr)

                _notify_progress(active_options.progress_callback, "running", state)
    finally:
        _close_handle(json_handle)
        _close_handle(db_conn)


def _normalize_filter(marker_filter: str | None) -> str | None:
    if marker_filter is None:
        return "all"
    if marker_filter in _VALID_FILTERS:
        return marker_filter
    upper_filter = marker_filter.upper()
    if upper_filter in {AD_START, AD_END, UNKNOWN}:
        return upper_filter
    lower_filter = marker_filter.lower()
    if lower_filter in {"all", "ad"}:
        return lower_filter
    return None


def _open_json_out(json_out: str | PathLike[str] | None) -> TextIO | None:
    if json_out is None:
        return None
    return Path(json_out).open("w", encoding="utf-8")


def _open_db(db_path: str | PathLike[str] | None) -> object | None:
    if db_path is None:
        return None
    return db.initialize_db(db_path)


def _iter_marker_source(marker_source: Iterable[AdMarker] | Callable[[], Iterable[AdMarker]]) -> Iterator[AdMarker]:
    source = marker_source() if callable(marker_source) else marker_source
    return iter(source)


def _timed_out(options: MonitorOptions, start_time: float) -> bool:
    if options.timeout is None:
        return False
    return options.clock() - start_time >= options.timeout


def _remaining_timeout(options: MonitorOptions, start_time: float) -> float | None:
    if options.timeout is None:
        return None
    return max(0.0, options.timeout - (options.clock() - start_time))


def _datetime_from_seconds(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)


def _maybe_retry_source_failure(
    exc: Exception,
    attempt: int,
    state: _MonitorState,
    options: MonitorOptions,
    start_time: float,
) -> _RetryOutcome:
    if not _is_retryable_source_error(exc, options):
        return _RetryOutcome(retried=False)

    policy = options.retry_policy or RetryPolicy()
    observed_now = options.clock()
    decision = policy.decision_for_attempt(attempt, now=_datetime_from_seconds(observed_now))
    redacted_error = _source_error_message(exc, fallback="marker iterator failed")
    if decision is None:
        return _RetryOutcome(retried=False, terminal_error=redacted_error)

    remaining = None if options.timeout is None else max(0.0, options.timeout - (observed_now - start_time))
    if remaining is not None and remaining <= 0:
        return _RetryOutcome(retried=False, terminal_reason="timeout")

    sleep_seconds = decision.delay_seconds if remaining is None else min(decision.delay_seconds, remaining)
    _notify_progress(
        options.progress_callback,
        "retrying",
        state,
        error=redacted_error,
        retry_attempt=decision.attempt,
        delay_seconds=decision.delay_seconds,
        next_retry_at=decision.next_retry_at,
    )
    try:
        options.retry_sleep(sleep_seconds)
    except Exception:
        return _RetryOutcome(retried=False, terminal_error="retry sleep failed")

    if remaining is not None and decision.delay_seconds > remaining:
        return _RetryOutcome(retried=False, terminal_reason="timeout")
    return _RetryOutcome(retried=True)


def _is_retryable_source_error(exc: Exception, options: MonitorOptions) -> bool:
    if not isinstance(exc, MonitorSourceError):
        return False
    if exc.phase not in {"setup", "iteration"}:
        return False
    if exc.stream_type in _RETRYABLE_STREAM_TYPES:
        return True
    if exc.stream_type is StreamType.MPEGTS:
        return urlparse(str(options.source_url)).scheme.lower() in {"http", "https"}
    return False


def _source_error_message(exc: Exception, *, fallback: str) -> str:
    if isinstance(exc, MonitorSourceError):
        message = _redact_monitor_error(str(exc))
        return message or fallback
    return fallback


def _redact_monitor_error(message: str) -> str:
    redacted = _PAYLOAD_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", redact_source_label(message))
    return _EMBEDDED_PATH_RE.sub(lambda match: Path(match.group(0)).name or "[redacted-path]", redacted)


def _matches_filter(marker: AdMarker, marker_filter: str) -> bool:
    if marker_filter == "all":
        return True
    if marker_filter == "ad":
        return marker.classification in {AD_START, AD_END}
    if marker_filter in _MARKER_TYPE_FILTERS:
        return marker.type.lower() == marker_filter
    return marker.classification == marker_filter


def _fatal_result(
    message: str,
    state: _MonitorState,
    stderr: TextIO,
    *,
    emit_summary: bool,
    progress_callback: MonitorProgressCallback | None = None,
) -> MonitorResult:
    _error(message, stderr)
    return _finish("error", state, stderr, error=message, emit_summary=emit_summary, progress_callback=progress_callback)


def _finish(
    reason: MonitorReason,
    state: _MonitorState,
    stderr: TextIO,
    *,
    error: str | None = None,
    emit_summary: bool,
    progress_callback: MonitorProgressCallback | None = None,
) -> MonitorResult:
    phase: MonitorPhase = "error" if reason == "error" else "completed"
    _notify_progress(progress_callback, phase, state, reason=reason, error=error)
    if emit_summary:
        stderr.write(
            "[tidemark] completed: "
            f"reason={reason} markers={state.markers_seen} "
            f"emitted={state.markers_emitted} filtered={state.markers_filtered}\n"
        )
    return MonitorResult(
        reason=reason,
        markers_seen=state.markers_seen,
        markers_emitted=state.markers_emitted,
        markers_filtered=state.markers_filtered,
        sink_warnings=state.sink_warnings,
        error=error,
    )


def _notify_progress(
    progress_callback: MonitorProgressCallback | None,
    phase: MonitorPhase,
    state: _MonitorState,
    *,
    reason: MonitorReason | None = None,
    error: str | None = None,
    retry_attempt: int | None = None,
    delay_seconds: float | None = None,
    next_retry_at: datetime | None = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(
            MonitorProgress(
                phase=phase,
                reason=reason,
                error=redact_source_label(error) if error is not None else None,
                retry_attempt=retry_attempt,
                delay_seconds=delay_seconds,
                next_retry_at=next_retry_at,
                counters={
                    "markers_seen": state.markers_seen,
                    "markers_emitted": state.markers_emitted,
                    "markers_filtered": state.markers_filtered,
                    "sink_warnings": state.sink_warnings,
                },
            )
        )
    except Exception:
        pass


def _error(message: str, stderr: TextIO) -> None:
    stderr.write(f"[tidemark] error: {message}\n")


def _warn(message: str, stderr: TextIO) -> None:
    stderr.write(f"[tidemark] warning: {message}\n")


def _close_handle(handle: object | None) -> None:
    if handle is None:
        return
    close = getattr(handle, "close", None)
    if callable(close):
        close()


__all__ = [
    "MarkerFilter",
    "MonitorOptions",
    "MonitorPhase",
    "MonitorProgress",
    "MonitorProgressCallback",
    "MonitorReason",
    "MonitorResult",
    "run_monitor",
]
