"""Best-effort structured runtime lifecycle JSONL logging.

Lifecycle logs are intentionally separate from marker NDJSON stdout.  This
module writes compact, append-only JSON objects to a configured file while
redacting source and error labels through the same status-safe helper used by
runtime health records.  Filesystem failures are reported to callers without
raising so monitor exit and marker semantics are not affected by logging.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tidemark.config import TidemarkConfig, default_runtime_dir
from tidemark.runtime.health import ReadDiagnostic, WriteResult, redact_source_label

NowFn = Callable[[], datetime]

_EMBEDDED_PATH_RE = re.compile(r"(?<![A-Za-z0-9+.-])(?:~|/|\.\.?/)[^\s]+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class LifecycleLogger:
    """Append compact lifecycle JSON events to one best-effort JSONL file."""

    path: Path
    now: NowFn = _utc_now

    def write(
        self,
        *,
        event: str,
        command: str,
        run_id: str,
        source_label: object,
        phase: str,
        counters: dict[str, object] | None = None,
        retry_attempt: int | None = None,
        next_retry_at: datetime | None = None,
        latency_ms: int | float | None = None,
        error: object | None = None,
        terminal_reason: object | None = None,
    ) -> WriteResult:
        """Append one lifecycle event and return a best-effort write result.

        Unsupported optional values are omitted before serialization.  Source,
        error, and terminal labels are redacted before JSON encoding; raw
        exception objects are converted with ``str()`` and never serialized with
        tracebacks or attributes.
        """
        path = Path(self.path)
        try:
            payload = _build_lifecycle_payload(
                event=event,
                command=command,
                run_id=run_id,
                source_label=source_label,
                phase=phase,
                counters=counters,
                retry_attempt=retry_attempt,
                next_retry_at=next_retry_at,
                latency_ms=latency_ms,
                error=error,
                terminal_reason=terminal_reason,
                now=self.now,
            )
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
        except (OSError, TypeError, ValueError):
            return WriteResult(
                ok=False,
                path=path,
                diagnostic=ReadDiagnostic(file_name=path.name, message="lifecycle log write failed"),
            )
        return WriteResult(ok=True, path=path)


def write_lifecycle_event(
    path: Path,
    *,
    event: str,
    command: str,
    run_id: str,
    source_label: object,
    phase: str,
    counters: dict[str, object] | None = None,
    retry_attempt: int | None = None,
    next_retry_at: datetime | None = None,
    latency_ms: int | float | None = None,
    error: object | None = None,
    terminal_reason: object | None = None,
    now: NowFn = _utc_now,
) -> WriteResult:
    """Write one lifecycle event to ``path`` without raising on log failures."""
    return LifecycleLogger(Path(path), now=now).write(
        event=event,
        command=command,
        run_id=run_id,
        source_label=source_label,
        phase=phase,
        counters=counters,
        retry_attempt=retry_attempt,
        next_retry_at=next_retry_at,
        latency_ms=latency_ms,
        error=error,
        terminal_reason=terminal_reason,
    )


def resolve_lifecycle_log_path(config: TidemarkConfig) -> Path:
    """Return the configured lifecycle log path or the runtime-dir default."""
    if config.paths.log_file is not None:
        return Path(config.paths.log_file)
    runtime_dir = Path(config.paths.runtime_dir) if config.paths.runtime_dir is not None else default_runtime_dir()
    return runtime_dir / "logs" / "tidemark.jsonl"


def _build_lifecycle_payload(
    *,
    event: str,
    command: str,
    run_id: str,
    source_label: object,
    phase: str,
    counters: dict[str, object] | None,
    retry_attempt: int | None,
    next_retry_at: datetime | None,
    latency_ms: int | float | None,
    error: object | None,
    terminal_reason: object | None,
    now: NowFn,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ts": _format_timestamp(now()),
        "event": str(event),
        "command": str(command),
        "run_id": str(run_id),
        "source_label": _redact_lifecycle_field(source_label),
        "phase": str(phase),
    }

    compact_counters = _compact_counters(counters)
    if compact_counters:
        payload["counters"] = compact_counters
    if _is_non_negative_int(retry_attempt):
        payload["retry_attempt"] = retry_attempt
    if isinstance(next_retry_at, datetime):
        payload["next_retry_at"] = _format_timestamp(next_retry_at)
    if _is_number(latency_ms):
        payload["latency_ms"] = latency_ms
    if error is not None:
        payload["error"] = _redact_lifecycle_field(error)
    if terminal_reason is not None:
        payload["terminal_reason"] = _redact_lifecycle_field(terminal_reason)
    return payload


def _redact_lifecycle_field(value: object) -> str:
    redacted = redact_source_label(str(value))
    return _EMBEDDED_PATH_RE.sub(lambda match: Path(match.group(0)).name or "[redacted-path]", redacted)


def _compact_counters(counters: dict[str, object] | None) -> dict[str, int | float]:
    if not isinstance(counters, dict):
        return {}
    compact: dict[str, int | float] = {}
    for key, value in counters.items():
        if not isinstance(key, str):
            continue
        if _is_number(value):
            compact[key] = value
    return compact


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


__all__ = [
    "LifecycleLogger",
    "resolve_lifecycle_log_path",
    "write_lifecycle_event",
]
