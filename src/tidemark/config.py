"""Library-first configuration loading and resolution for tidemark.

This module intentionally has no CLI framework dependency.  It provides a
small typed boundary that future command wrappers can consume while preserving
current command defaults when no TOML, environment, or explicit CLI override is
provided.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_ALLOWED_STREAM_TYPES = frozenset({"auto", "hls", "icecast", "icy", "mpegts", "udp"})
_SECRET_FIELD_TOKENS = ("api_key", "token", "secret", "password")
_URL_FIELD_TOKENS = ("url", "uri")
_PATH_FIELD_TOKENS = ("path", "dir", "file", "db")


class ConfigError(ValueError):
    """Base class for redacted config diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        field_path: str | None = None,
        source: str | None = None,
        value: object | None = None,
    ) -> None:
        self.field_path = field_path
        self.source = source
        self.value = value
        parts = [message]
        if field_path is not None:
            parts.append(f"field={field_path}")
        if source is not None:
            parts.append(f"source={source}")
        super().__init__("; ".join(parts))


class ConfigFileError(ConfigError):
    """Raised for config file discovery, read, or TOML parse failures."""


class ConfigValidationError(ConfigError):
    """Raised when a known config source contains an invalid value."""


@dataclass(frozen=True)
class PathsConfig:
    db: Path = Path("tidemark.db")
    runtime_dir: Path | None = None
    log_file: Path | None = None
    retention_dir: Path | None = None


@dataclass(frozen=True)
class MonitorConfig:
    stream_type: str = "auto"
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class IngestConfig:
    include_manifest_markers: bool = True
    fingerprint: bool = False


@dataclass(frozen=True)
class SearchConfig:
    context_seconds: float = 5.0


@dataclass(frozen=True)
class ReportConfig:
    min_score: float = 0.8
    min_count: int = 2


@dataclass(frozen=True)
class FingerprintConfig:
    enabled: bool = False
    api_key: str | None = None
    lookup_timeout_seconds: float | None = None
    retry_attempts: int = 0
    retry_backoff_seconds: float = 0.0


@dataclass(frozen=True)
class TidemarkConfig:
    paths: PathsConfig = PathsConfig()
    monitor: MonitorConfig = MonitorConfig()
    ingest: IngestConfig = IngestConfig()
    search: SearchConfig = SearchConfig()
    report: ReportConfig = ReportConfig()
    fingerprint: FingerprintConfig = FingerprintConfig()


@dataclass(frozen=True)
class MonitorOverrides:
    db_path: Path | None = None
    stream_type: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class MonitorOptions:
    db_path: Path
    stream_type: str
    timeout_seconds: float | None


@dataclass(frozen=True)
class IngestOverrides:
    db_path: Path | None = None
    include_manifest_markers: bool | None = None
    fingerprint: bool | None = None
    acoustid_api_key: str | None = None
    lookup_timeout_seconds: float | None = None


@dataclass(frozen=True)
class IngestOptions:
    db_path: Path
    include_manifest_markers: bool
    fingerprint: bool
    acoustid_api_key: str | None
    lookup_timeout_seconds: float | None


@dataclass(frozen=True)
class SearchOverrides:
    db_path: Path | None = None
    context_seconds: float | None = None


@dataclass(frozen=True)
class SearchOptions:
    db_path: Path
    context_seconds: float


@dataclass(frozen=True)
class ReportOverrides:
    db_path: Path | None = None
    min_score: float | None = None
    min_count: int | None = None


@dataclass(frozen=True)
class ReportOptions:
    db_path: Path
    min_score: float
    min_count: int


@dataclass(frozen=True)
class ClipOverrides:
    db_path: Path | None = None


@dataclass(frozen=True)
class ClipOptions:
    db_path: Path


_SECTION_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "paths": frozenset({"db", "runtime_dir", "log_file", "retention_dir"}),
        "monitor": frozenset({"stream_type", "timeout_seconds"}),
        "ingest": frozenset({"include_manifest_markers", "fingerprint"}),
        "search": frozenset({"context_seconds"}),
        "report": frozenset({"min_score", "min_count"}),
        "fingerprint": frozenset(
            {"enabled", "api_key", "lookup_timeout_seconds", "retry_attempts", "retry_backoff_seconds"}
        ),
    }
)

_ENV_FIELDS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "TIDEMARK_DB": ("paths", "db"),
        "TIDEMARK_RUNTIME_DIR": ("paths", "runtime_dir"),
        "TIDEMARK_LOG_FILE": ("paths", "log_file"),
        "TIDEMARK_RETENTION_DIR": ("paths", "retention_dir"),
        "TIDEMARK_MONITOR_STREAM_TYPE": ("monitor", "stream_type"),
        "TIDEMARK_MONITOR_TIMEOUT_SECONDS": ("monitor", "timeout_seconds"),
        "TIDEMARK_INGEST_FINGERPRINT": ("ingest", "fingerprint"),
        "TIDEMARK_LOOKUP_TIMEOUT_SECONDS": ("fingerprint", "lookup_timeout_seconds"),
        "ACOUSTID_API_KEY": ("fingerprint", "api_key"),
    }
)


def default_config_path() -> Path:
    """Return the conventional per-user TOML config path."""
    return Path("~/.config/tidemark/tidemark.toml").expanduser()


def default_runtime_dir() -> Path:
    """Return the conventional per-user runtime directory."""
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "tidemark"
    return Path("~/.local/state/tidemark").expanduser()


def load_config(
    path: Path | None = None,
    *,
    explicit: bool = False,
    env: Mapping[str, str] = os.environ,
) -> TidemarkConfig:
    """Load TOML config, then overlay environment values.

    Precedence is built-in defaults < config file < environment.  CLI values are
    intentionally not accepted here; they are applied by command-specific
    resolver functions so wrappers can preserve the CLI > env > config > default
    order deterministically.
    """
    config = TidemarkConfig()
    resolved_path, should_require = _selected_config_path(path, explicit=explicit, env=env)

    if resolved_path.exists():
        config = _merge_toml_config(config, resolved_path)
    elif should_require:
        raise ConfigFileError(
            f"config file not found: {redact_path(resolved_path)}",
            source="config",
        )

    return _merge_env_config(config, env)


def resolve_monitor_options(config: TidemarkConfig, overrides: MonitorOverrides | None = None) -> MonitorOptions:
    overrides = overrides or MonitorOverrides()
    stream_type = _coalesce(overrides.stream_type, config.monitor.stream_type)
    timeout_seconds = _coalesce(overrides.timeout_seconds, config.monitor.timeout_seconds)
    return MonitorOptions(
        db_path=_coalesce(overrides.db_path, config.paths.db),
        stream_type=_validate_stream_type(stream_type, field_path="monitor.stream_type", source="CLI"),
        timeout_seconds=_validate_optional_non_negative_float(
            timeout_seconds,
            field_path="monitor.timeout_seconds",
            source="CLI" if overrides.timeout_seconds is not None else "config",
        ),
    )


def resolve_ingest_options(config: TidemarkConfig, overrides: IngestOverrides | None = None) -> IngestOptions:
    overrides = overrides or IngestOverrides()
    fingerprint = _coalesce(overrides.fingerprint, config.ingest.fingerprint)
    return IngestOptions(
        db_path=_coalesce(overrides.db_path, config.paths.db),
        include_manifest_markers=_coalesce(overrides.include_manifest_markers, config.ingest.include_manifest_markers),
        fingerprint=fingerprint,
        acoustid_api_key=_coalesce(overrides.acoustid_api_key, config.fingerprint.api_key),
        lookup_timeout_seconds=_validate_optional_non_negative_float(
            _coalesce(overrides.lookup_timeout_seconds, config.fingerprint.lookup_timeout_seconds),
            field_path="fingerprint.lookup_timeout_seconds",
            source="CLI" if overrides.lookup_timeout_seconds is not None else "config",
        ),
    )


def resolve_search_options(config: TidemarkConfig, overrides: SearchOverrides | None = None) -> SearchOptions:
    overrides = overrides or SearchOverrides()
    context_seconds = _coalesce(overrides.context_seconds, config.search.context_seconds)
    return SearchOptions(
        db_path=_coalesce(overrides.db_path, config.paths.db),
        context_seconds=_validate_non_negative_float(
            context_seconds,
            field_path="search.context_seconds",
            source="CLI" if overrides.context_seconds is not None else "config",
        ),
    )


def resolve_report_options(config: TidemarkConfig, overrides: ReportOverrides | None = None) -> ReportOptions:
    overrides = overrides or ReportOverrides()
    min_score = _coalesce(overrides.min_score, config.report.min_score)
    min_count = _coalesce(overrides.min_count, config.report.min_count)
    return ReportOptions(
        db_path=_coalesce(overrides.db_path, config.paths.db),
        min_score=_validate_score(min_score, field_path="report.min_score", source="CLI"),
        min_count=_validate_positive_int(min_count, field_path="report.min_count", source="CLI"),
    )


def resolve_clip_options(config: TidemarkConfig, overrides: ClipOverrides | None = None) -> ClipOptions:
    overrides = overrides or ClipOverrides()
    return ClipOptions(db_path=_coalesce(overrides.db_path, config.paths.db))


def redact_value(field_path: str, value: object) -> str:
    """Return a safe display placeholder for a potentially sensitive value."""
    lowered = field_path.lower()
    if any(token in lowered for token in _SECRET_FIELD_TOKENS):
        return "[redacted]"
    if any(token in lowered for token in _URL_FIELD_TOKENS):
        return "[redacted-url]"
    if any(token in lowered for token in _PATH_FIELD_TOKENS):
        return "[redacted-path]"
    return "[redacted-value]"


def redact_path(path: Path) -> str:
    """Return a localizable path label without leaking private parent dirs."""
    return path.name or "[redacted-path]"


def _selected_config_path(
    path: Path | None,
    *,
    explicit: bool,
    env: Mapping[str, str],
) -> tuple[Path, bool]:
    if path is not None:
        return Path(path).expanduser(), explicit
    env_path = env.get("TIDEMARK_CONFIG")
    if env_path:
        return Path(env_path).expanduser(), True
    return default_config_path(), False


def _merge_toml_config(config: TidemarkConfig, path: Path) -> TidemarkConfig:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigFileError(
            f"TOML parse failed in {redact_path(path)}",
            source="config",
        ) from exc
    except OSError as exc:
        raise ConfigFileError(
            f"config file could not be read: {redact_path(path)}",
            source="config",
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigValidationError("config root must be a table", source="config")

    for section, values in raw.items():
        if section not in _SECTION_FIELDS:
            raise ConfigValidationError(
                "unknown config section",
                field_path=str(section),
                source="config",
            )
        if not isinstance(values, dict):
            raise ConfigValidationError(
                "config section must be a table",
                field_path=str(section),
                source="config",
            )
        for field, value in values.items():
            if field not in _SECTION_FIELDS[section]:
                raise ConfigValidationError(
                    "unknown config field",
                    field_path=f"{section}.{field}",
                    source="config",
                    value=value,
                )
            config = _apply_value(config, section, field, value, source="config")
    return config


def _merge_env_config(config: TidemarkConfig, env: Mapping[str, str]) -> TidemarkConfig:
    for env_name, (section, field) in _ENV_FIELDS.items():
        if env_name not in env:
            continue
        config = _apply_value(config, section, field, env[env_name], source=env_name)
    return config


def _apply_value(config: TidemarkConfig, section: str, field: str, value: object, *, source: str) -> TidemarkConfig:
    field_path = f"{section}.{field}"
    if section == "paths":
        parsed = _parse_path(value, field_path=field_path, source=source)
        return replace(config, paths=replace(config.paths, **{field: parsed}))
    if section == "monitor":
        if field == "stream_type":
            parsed = _validate_stream_type(value, field_path=field_path, source=source)
        else:
            parsed = _parse_optional_non_negative_float(value, field_path=field_path, source=source)
        return replace(config, monitor=replace(config.monitor, **{field: parsed}))
    if section == "ingest":
        parsed = _parse_bool(value, field_path=field_path, source=source)
        return replace(config, ingest=replace(config.ingest, **{field: parsed}))
    if section == "search":
        parsed = _parse_non_negative_float(value, field_path=field_path, source=source)
        return replace(config, search=replace(config.search, **{field: parsed}))
    if section == "report":
        if field == "min_score":
            parsed = _parse_score(value, field_path=field_path, source=source)
        else:
            parsed = _parse_positive_int(value, field_path=field_path, source=source)
        return replace(config, report=replace(config.report, **{field: parsed}))
    if section == "fingerprint":
        if field == "enabled":
            parsed = _parse_bool(value, field_path=field_path, source=source)
        elif field == "api_key":
            parsed = _parse_optional_secret(value, field_path=field_path, source=source)
        elif field in {"lookup_timeout_seconds", "retry_backoff_seconds"}:
            parsed = _parse_optional_non_negative_float(value, field_path=field_path, source=source)
            if field == "retry_backoff_seconds" and parsed is None:
                parsed = 0.0
        else:
            parsed = _parse_non_negative_int(value, field_path=field_path, source=source)
        return replace(config, fingerprint=replace(config.fingerprint, **{field: parsed}))
    raise AssertionError(f"unknown config section: {section}")


def _parse_path(value: object, *, field_path: str, source: str) -> Path:
    if not isinstance(value, str) or value.strip() == "":
        _invalid("path must be a non-empty string", field_path=field_path, source=source, value=value)
    return Path(value).expanduser()


def _parse_bool(value: object, *, field_path: str, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    _invalid("boolean value is invalid", field_path=field_path, source=source, value=value)


def _parse_optional_secret(value: object, *, field_path: str, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _invalid("secret value must be a string", field_path=field_path, source=source, value=value)
    text = value.strip()
    return text or None


def _parse_optional_non_negative_float(value: object, *, field_path: str, source: str) -> float | None:
    if value is None:
        return None
    return _parse_non_negative_float(value, field_path=field_path, source=source)


def _parse_non_negative_float(value: object, *, field_path: str, source: str) -> float:
    parsed = _parse_float(value, field_path=field_path, source=source)
    return _validate_non_negative_float(parsed, field_path=field_path, source=source)


def _parse_float(value: object, *, field_path: str, source: str) -> float:
    if isinstance(value, bool):
        _invalid("number must be a float", field_path=field_path, source=source, value=value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            _invalid("number is invalid", field_path=field_path, source=source, value=value)
    _invalid("number must be a float", field_path=field_path, source=source, value=value)


def _parse_score(value: object, *, field_path: str, source: str) -> float:
    return _validate_score(_parse_float(value, field_path=field_path, source=source), field_path=field_path, source=source)


def _parse_positive_int(value: object, *, field_path: str, source: str) -> int:
    parsed = _parse_int(value, field_path=field_path, source=source)
    return _validate_positive_int(parsed, field_path=field_path, source=source)


def _parse_non_negative_int(value: object, *, field_path: str, source: str) -> int:
    parsed = _parse_int(value, field_path=field_path, source=source)
    if parsed < 0:
        _invalid("integer must be >= 0", field_path=field_path, source=source, value=value)
    return parsed


def _parse_int(value: object, *, field_path: str, source: str) -> int:
    if isinstance(value, bool):
        _invalid("integer must be an int", field_path=field_path, source=source, value=value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            _invalid("integer is invalid", field_path=field_path, source=source, value=value)
    _invalid("integer must be an int", field_path=field_path, source=source, value=value)


def _validate_stream_type(value: object, *, field_path: str, source: str) -> str:
    if not isinstance(value, str):
        _invalid("stream type must be a string", field_path=field_path, source=source, value=value)
    normalized = value.strip().lower()
    if normalized not in _ALLOWED_STREAM_TYPES:
        _invalid("stream type is invalid", field_path=field_path, source=source, value=value)
    return normalized


def _validate_optional_non_negative_float(value: object, *, field_path: str, source: str) -> float | None:
    if value is None:
        return None
    return _validate_non_negative_float(value, field_path=field_path, source=source)


def _validate_non_negative_float(value: object, *, field_path: str, source: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _invalid("number must be a float", field_path=field_path, source=source, value=value)
    parsed = float(value)
    if parsed < 0:
        _invalid("number must be >= 0", field_path=field_path, source=source, value=value)
    return parsed


def _validate_score(value: object, *, field_path: str, source: str) -> float:
    parsed = _validate_non_negative_float(value, field_path=field_path, source=source)
    if parsed > 1:
        _invalid("score must be between 0 and 1", field_path=field_path, source=source, value=value)
    return parsed


def _validate_positive_int(value: object, *, field_path: str, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _invalid("integer must be an int", field_path=field_path, source=source, value=value)
    if value < 1:
        _invalid("integer must be >= 1", field_path=field_path, source=source, value=value)
    return value


def _invalid(message: str, *, field_path: str, source: str, value: object) -> Any:
    safe_value = redact_value(field_path, value)
    raise ConfigValidationError(
        f"{message}; value={safe_value}",
        field_path=field_path,
        source=source,
        value=value,
    )


def _coalesce[T](first: T | None, fallback: T) -> T:
    return fallback if first is None else first


__all__ = [
    "ClipOptions",
    "ClipOverrides",
    "ConfigError",
    "ConfigFileError",
    "ConfigValidationError",
    "FingerprintConfig",
    "IngestConfig",
    "IngestOptions",
    "IngestOverrides",
    "MonitorConfig",
    "MonitorOptions",
    "MonitorOverrides",
    "PathsConfig",
    "ReportConfig",
    "ReportOptions",
    "ReportOverrides",
    "SearchConfig",
    "SearchOptions",
    "SearchOverrides",
    "TidemarkConfig",
    "default_config_path",
    "default_runtime_dir",
    "load_config",
    "redact_path",
    "redact_value",
    "resolve_clip_options",
    "resolve_ingest_options",
    "resolve_monitor_options",
    "resolve_report_options",
    "resolve_search_options",
]
