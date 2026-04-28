"""Side-effect-safe runtime health records for long-running tidemark commands.

This module intentionally has no CLI or SQLite dependency.  Writers publish one
compact JSON file per run under ``<runtime_dir>/runs/``; readers scan that
directory without creating it and return malformed-file diagnostics instead of
raising so status surfaces remain safe under partial filesystem failure.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from tidemark.config import redact_path

SCHEMA_VERSION = 1
DEFAULT_STALE_AFTER_SECONDS = 120

StatusLabel = Literal["active", "stale", "not-running"]
NowFn = Callable[[], datetime]
PidExistsFn = Callable[[int], bool]

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|fingerprint)\s*[=:]\s*[^\s&]+"
)
_SECRET_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[a-z0-9._~+/=-]+")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "command",
        "source_label",
        "pid",
        "started_at",
        "heartbeat_at",
        "phase",
        "counters",
        "retry",
        "terminal",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is invalid")
    normalized = value.replace("Z", "+00:00")
    try:
        return _as_utc(datetime.fromisoformat(normalized))
    except ValueError as exc:
        raise ValueError("timestamp is invalid") from exc


def _safe_identifier(value: str, *, fallback: str = "run", lowercase: bool = True) -> str:
    normalized = _SAFE_ID_RE.sub("-", value.strip()).strip("-._")
    if lowercase:
        normalized = normalized.lower()
    return normalized or fallback


def make_run_id(
    command: str,
    *,
    now: datetime | None = None,
    pid: int | None = None,
    entropy: str | None = None,
) -> str:
    """Return a stable, filename-safe run id with command, timestamp, pid, entropy."""
    timestamp = _format_timestamp(now or _utc_now()).replace("-", "").replace(":", "")
    safe_entropy = _safe_identifier(entropy or secrets.token_hex(4), fallback="run")
    safe_pid = str(pid if pid is not None else os.getpid())
    return "-".join([_safe_identifier(command, fallback="tidemark"), timestamp, safe_pid, safe_entropy])


def redact_source_label(value: object) -> str:
    """Return a status-safe source/error label without secrets or private dirs."""
    if value is None:
        return ""
    if isinstance(value, PathLike):
        return "[local file]"

    text = str(value)
    if not text:
        return ""

    stripped = text.strip()
    parsed = urlsplit(stripped)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        host = parsed.hostname or "[redacted-url]"
        path = parsed.path or ""
        return f"{host}{path}" or "[redacted-url]"
    if parsed.scheme == "file":
        return "[local file]"

    path = Path(stripped).expanduser()
    if _looks_like_path(stripped, path):
        return "[local file]"

    redacted = _URL_RE.sub(lambda match: redact_source_label(match.group(0)), stripped)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", redacted)
    redacted = _SECRET_BEARER_RE.sub(lambda match: f"{match.group(1)} [redacted]", redacted)
    return redacted


def _looks_like_path(text: str, path: Path) -> bool:
    if any(character.isspace() for character in text):
        return False
    return text.startswith(("/", "~", "./", "../")) or "\\" in text


@dataclass(frozen=True)
class RetryState:
    """Retry scheduling state for status/observability surfaces."""

    attempt: int = 0
    next_retry_at: datetime | None = None
    last_retry_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 0:
            raise ValueError("retry.attempt must be a non-negative integer")
        if self.next_retry_at is not None and not isinstance(self.next_retry_at, datetime):
            raise ValueError("retry.next_retry_at must be a datetime")
        if self.next_retry_at is not None:
            object.__setattr__(self, "next_retry_at", _as_utc(self.next_retry_at))
        if self.last_retry_error is not None:
            object.__setattr__(self, "last_retry_error", redact_source_label(self.last_retry_error))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "next_retry_at": _format_timestamp(self.next_retry_at) if self.next_retry_at else None,
            "last_retry_error": redact_source_label(self.last_retry_error) if self.last_retry_error else None,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RetryState":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("retry must be an object")
        attempt = payload.get("attempt", 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError("retry.attempt must be a non-negative integer")
        next_retry_at = payload.get("next_retry_at")
        return cls(
            attempt=attempt,
            next_retry_at=_parse_timestamp(next_retry_at) if next_retry_at is not None else None,
            last_retry_error=redact_source_label(payload.get("last_retry_error"))
            if payload.get("last_retry_error") is not None
            else None,
        )


@dataclass(frozen=True)
class HealthRecord:
    """Serializable runtime status for one command run."""

    run_id: str
    command: str
    source_label: str
    pid: int
    started_at: datetime
    heartbeat_at: datetime
    phase: str
    counters: dict[str, int | float] = field(default_factory=dict)
    retry: RetryState = field(default_factory=RetryState)
    last_error: str | None = None
    terminal: bool = False
    terminal_reason: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "command": self.command,
            "source_label": redact_source_label(self.source_label),
            "pid": self.pid,
            "started_at": _format_timestamp(self.started_at),
            "heartbeat_at": _format_timestamp(self.heartbeat_at),
            "phase": self.phase,
            "counters": dict(self.counters),
            "retry": self.retry.to_json_dict(),
            "last_error": redact_source_label(self.last_error) if self.last_error else None,
            "terminal": self.terminal,
            "terminal_reason": redact_source_label(self.terminal_reason) if self.terminal_reason else None,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "HealthRecord":
        if not isinstance(payload, dict):
            raise ValueError("status file root must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema version is unsupported")
        missing = sorted(_REQUIRED_FIELDS - payload.keys())
        if missing:
            raise ValueError(f"missing required field: {missing[0]}")

        counters = payload.get("counters")
        if not isinstance(counters, dict):
            raise ValueError("counters must be an object")
        parsed_counters: dict[str, int | float] = {}
        for key, value in counters.items():
            if not isinstance(key, str):
                raise ValueError("counter names must be strings")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("counter values must be numeric")
            parsed_counters[key] = value

        pid = payload.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise ValueError("pid must be an integer")
        terminal = payload.get("terminal")
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be a boolean")

        return cls(
            schema_version=SCHEMA_VERSION,
            run_id=_required_str(payload, "run_id"),
            command=_required_str(payload, "command"),
            source_label=redact_source_label(_required_str(payload, "source_label")),
            pid=pid,
            started_at=_parse_timestamp(payload.get("started_at")),
            heartbeat_at=_parse_timestamp(payload.get("heartbeat_at")),
            phase=_required_str(payload, "phase"),
            counters=parsed_counters,
            retry=RetryState.from_json_dict(payload.get("retry")),
            last_error=redact_source_label(payload.get("last_error")) if payload.get("last_error") is not None else None,
            terminal=terminal,
            terminal_reason=redact_source_label(payload.get("terminal_reason"))
            if payload.get("terminal_reason") is not None
            else None,
        )


def _required_str(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ReadDiagnostic:
    """Redacted, status-safe diagnostic for malformed health files."""

    file_name: str
    message: str


@dataclass(frozen=True)
class WriteResult:
    """Best-effort write result returned by reporter methods."""

    ok: bool
    path: Path
    diagnostic: ReadDiagnostic | None = None


@dataclass(frozen=True)
class StatusEntry:
    """A health record plus deterministic status classification."""

    record: HealthRecord
    status: StatusLabel
    heartbeat_age_seconds: float
    diagnostics: tuple[ReadDiagnostic, ...] = ()


@dataclass
class HealthReporter:
    """Best-effort atomic writer for one runtime health record."""

    runtime_dir: Path
    record: HealthRecord
    now: NowFn = _utc_now

    @property
    def runs_dir(self) -> Path:
        return Path(self.runtime_dir) / "runs"

    @property
    def path(self) -> Path:
        return self.runs_dir / f"{_safe_identifier(self.record.run_id, lowercase=False)}.json"

    def start(self, *, phase: str = "starting", counters: dict[str, int | float] | None = None) -> WriteResult:
        current = _as_utc(self.now())
        self.record = replace(
            self.record,
            phase=phase,
            counters=_merge_counters(self.record.counters, counters),
            started_at=current,
            heartbeat_at=current,
            terminal=False,
            terminal_reason=None,
        )
        return self._write()

    def update(
        self,
        *,
        phase: str | None = None,
        counters: dict[str, int | float] | None = None,
        last_error: object | None = None,
    ) -> WriteResult:
        self.record = replace(
            self.record,
            heartbeat_at=_as_utc(self.now()),
            phase=phase or self.record.phase,
            counters=_merge_counters(self.record.counters, counters),
            last_error=redact_source_label(last_error) if last_error is not None else self.record.last_error,
        )
        return self._write()

    def retry(
        self,
        *,
        attempt: int,
        next_retry_at: datetime,
        error: object,
        phase: str = "retrying",
        counters: dict[str, int | float] | None = None,
    ) -> WriteResult:
        self.record = replace(
            self.record,
            heartbeat_at=_as_utc(self.now()),
            phase=phase,
            counters=_merge_counters(self.record.counters, counters),
            retry=RetryState(
                attempt=attempt,
                next_retry_at=next_retry_at,
                last_retry_error=redact_source_label(error),
            ),
            terminal=False,
            terminal_reason=None,
        )
        return self._write()

    def finish(
        self,
        *,
        reason: object = "finished",
        phase: str = "finished",
        counters: dict[str, int | float] | None = None,
    ) -> WriteResult:
        self.record = replace(
            self.record,
            heartbeat_at=_as_utc(self.now()),
            phase=phase,
            counters=_merge_counters(self.record.counters, counters),
            terminal=True,
            terminal_reason=redact_source_label(reason),
        )
        return self._write()

    def fail(
        self,
        error: object,
        *,
        phase: str = "failed",
        reason: object = "failed",
        counters: dict[str, int | float] | None = None,
    ) -> WriteResult:
        self.record = replace(
            self.record,
            heartbeat_at=_as_utc(self.now()),
            phase=phase,
            counters=_merge_counters(self.record.counters, counters),
            last_error=redact_source_label(error),
            terminal=True,
            terminal_reason=redact_source_label(reason),
        )
        return self._write()

    def _write(self) -> WriteResult:
        path = self.path
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.record.to_json_dict(), sort_keys=True, separators=(",", ":"))
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(path)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return WriteResult(
                ok=False,
                path=path,
                diagnostic=ReadDiagnostic(file_name=path.name, message="health write failed"),
            )
        return WriteResult(ok=True, path=path)


def create_reporter(
    runtime_dir: Path,
    *,
    command: str,
    source: object,
    run_id: str | None = None,
    pid: int | None = None,
    now: NowFn = _utc_now,
) -> HealthReporter:
    current = _as_utc(now())
    process_id = pid if pid is not None else os.getpid()
    safe_run_id = run_id or make_run_id(command, now=current, pid=process_id)
    return HealthReporter(
        runtime_dir=Path(runtime_dir),
        now=now,
        record=HealthRecord(
            run_id=safe_run_id,
            command=command,
            source_label=redact_source_label(source),
            pid=process_id,
            started_at=current,
            heartbeat_at=current,
            phase="initialized",
        ),
    )


def read_status_entries(
    runtime_dir: Path,
    *,
    now: datetime | None = None,
    pid_exists: PidExistsFn = lambda pid: globals()["pid_exists"](pid),
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> tuple[list[StatusEntry], list[ReadDiagnostic]]:
    """Read and classify all run records without creating runtime directories."""
    runs_dir = Path(runtime_dir) / "runs"
    if not runs_dir.exists():
        return [], []
    if not runs_dir.is_dir():
        return [], [ReadDiagnostic(file_name="runs", message="runs path is not a directory")]

    entries: list[StatusEntry] = []
    diagnostics: list[ReadDiagnostic] = []
    observed_now = _as_utc(now or _utc_now())
    for path in sorted(runs_dir.glob("*.json"), key=lambda item: item.name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = HealthRecord.from_json_dict(payload)
        except json.JSONDecodeError:
            diagnostics.append(ReadDiagnostic(file_name=path.name, message="invalid JSON in status file"))
            continue
        except OSError:
            diagnostics.append(ReadDiagnostic(file_name=path.name, message="status file could not be read"))
            continue
        except ValueError as exc:
            diagnostics.append(ReadDiagnostic(file_name=path.name, message=redact_source_label(str(exc))))
            continue
        entries.append(
            classify_record(
                record,
                now=observed_now,
                pid_exists=pid_exists,
                stale_after_seconds=stale_after_seconds,
            )
        )
    return entries, diagnostics


def classify_record(
    record: HealthRecord,
    *,
    now: datetime | None = None,
    pid_exists: PidExistsFn = lambda pid: globals()["pid_exists"](pid),
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> StatusEntry:
    observed_now = _as_utc(now or _utc_now())
    heartbeat_age = max(0.0, (observed_now - _as_utc(record.heartbeat_at)).total_seconds())
    diagnostics: list[ReadDiagnostic] = []

    if record.terminal:
        return StatusEntry(record=record, status="not-running", heartbeat_age_seconds=heartbeat_age)
    if not isinstance(record.pid, int) or isinstance(record.pid, bool) or record.pid <= 0:
        diagnostics.append(ReadDiagnostic(file_name=f"{_safe_identifier(record.run_id)}.json", message="invalid pid"))
        return StatusEntry(
            record=record,
            status="not-running",
            heartbeat_age_seconds=heartbeat_age,
            diagnostics=tuple(diagnostics),
        )

    try:
        live = pid_exists(record.pid)
    except OSError:
        diagnostics.append(
            ReadDiagnostic(file_name=f"{_safe_identifier(record.run_id)}.json", message="pid liveness probe failed")
        )
        status: StatusLabel = "stale" if heartbeat_age > stale_after_seconds else "not-running"
        return StatusEntry(record=record, status=status, heartbeat_age_seconds=heartbeat_age, diagnostics=tuple(diagnostics))

    if not live:
        status = "not-running"
    elif heartbeat_age > stale_after_seconds:
        status = "stale"
    else:
        status = "active"
    return StatusEntry(record=record, status=status, heartbeat_age_seconds=heartbeat_age, diagnostics=tuple(diagnostics))


def pid_exists(pid: int) -> bool:
    """Return whether a process appears alive, without raising for normal misses."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def format_status_report(
    entries: list[StatusEntry],
    diagnostics: list[ReadDiagnostic],
    *,
    runtime_dir: Path,
) -> str:
    """Return a compact, redacted status report for CLI and agent inspection."""
    lines = [f"Runtime directory: {redact_path(Path(runtime_dir))}"]
    if not entries:
        lines.append("No runtime health records found; tidemark is not running.")
    else:
        lines.append(f"Runs: {len(entries)}")
        for entry in entries:
            lines.append(_format_status_entry(entry))

    entry_diagnostics = [diagnostic for entry in entries for diagnostic in entry.diagnostics]
    all_diagnostics = [*diagnostics, *entry_diagnostics]
    if all_diagnostics:
        lines.append(f"Diagnostics: {len(all_diagnostics)} malformed status file(s) skipped")
        for diagnostic in all_diagnostics:
            lines.append(f"- {diagnostic.file_name}: {diagnostic.message}")
    return "\n".join(lines)


def _format_status_entry(entry: StatusEntry) -> str:
    record = entry.record
    fields = [
        f"run_id={record.run_id}",
        f"command={record.command}",
        f"pid={record.pid}",
        f"phase={record.phase}",
        f"state={entry.status}",
        f"heartbeat_age={int(entry.heartbeat_age_seconds)}s",
        f"source={record.source_label}",
        f"counters={_format_counters(record.counters)}",
        f"retry={_format_retry(record.retry)}",
    ]
    if record.last_error:
        fields.append(f"last_error={record.last_error}")
    if record.terminal:
        fields.append(f"terminal={record.terminal_reason or 'finished'}")
    return "- " + " ".join(fields)


def _format_counters(counters: dict[str, int | float]) -> str:
    if not counters:
        return "none"
    return ",".join(f"{key}={counters[key]:g}" for key in sorted(counters))


def _format_retry(retry: RetryState) -> str:
    parts = [f"attempt={retry.attempt}"]
    if retry.next_retry_at is not None:
        parts.append(f"next={_format_timestamp(retry.next_retry_at)}")
    if retry.last_retry_error:
        parts.append(f"last_error={retry.last_retry_error}")
    return ",".join(parts)


def _merge_counters(
    current: dict[str, int | float],
    updates: dict[str, int | float] | None,
) -> dict[str, int | float]:
    merged = dict(current)
    if not updates:
        return merged
    for key, value in updates.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        merged[key] = value
    return merged


__all__ = [
    "DEFAULT_STALE_AFTER_SECONDS",
    "SCHEMA_VERSION",
    "HealthRecord",
    "HealthReporter",
    "ReadDiagnostic",
    "RetryState",
    "StatusEntry",
    "WriteResult",
    "classify_record",
    "create_reporter",
    "format_status_report",
    "make_run_id",
    "pid_exists",
    "read_status_entries",
    "redact_source_label",
]
