from __future__ import annotations

import os
from pathlib import Path

import pytest

from tidemark.config import (
    ConfigError,
    IngestOverrides,
    MonitorOverrides,
    SearchOverrides,
    load_config,
    redact_value,
    resolve_clip_options,
    resolve_ingest_options,
    resolve_monitor_options,
    resolve_report_options,
    resolve_search_options,
)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tidemark.toml"
    path.write_text(body, encoding="utf-8")
    return path


def assert_safe_message(exc: BaseException, *raw_values: str) -> str:
    message = str(exc)
    for value in raw_values:
        assert value not in message
    return message


def test_missing_default_config_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(env={})

    assert config.paths.db == Path("tidemark.db")
    assert config.search.context_seconds == 5.0
    assert config.report.min_score == 0.8
    assert config.report.min_count == 2
    assert config.ingest.include_manifest_markers is True
    assert config.fingerprint.enabled is False
    assert config.monitor.stream_type == "auto"
    assert config.monitor.timeout_seconds is None


def test_explicit_missing_config_fails_with_safe_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"

    with pytest.raises(ConfigError) as caught:
        load_config(path, explicit=True, env={})

    message = str(caught.value)
    assert "config" in message
    assert "missing.toml" in message
    assert str(tmp_path) not in message


def test_malformed_toml_fails_without_source_line(tmp_path: Path) -> None:
    secret = "super-secret-api-key"
    path = write_config(tmp_path, f'[fingerprint\napi_key = "{secret}"\n')

    with pytest.raises(ConfigError) as caught:
        load_config(path, explicit=True, env={})

    message = assert_safe_message(caught.value, secret, "api_key =")
    assert "config" in message
    assert "TOML" in message


def test_unknown_sections_and_fields_fail_with_field_paths(tmp_path: Path) -> None:
    section_path = write_config(tmp_path, "[unknown]\nvalue = true\n")
    with pytest.raises(ConfigError) as section_error:
        load_config(section_path, explicit=True, env={})
    assert "unknown" in str(section_error.value)

    field_path = write_config(tmp_path, "[paths]\nunknown_field = true\n")
    with pytest.raises(ConfigError) as field_error:
        load_config(field_path, explicit=True, env={})
    assert "paths.unknown_field" in str(field_error.value)


def test_invalid_types_ranges_and_secret_fields_do_not_leak_values(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
[paths]
db = 42

[monitor]
timeout_seconds = -1

[fingerprint]
api_key = "raw-secret-key"
lookup_timeout_seconds = -0.5
""".strip(),
    )

    with pytest.raises(ConfigError) as caught:
        load_config(path, explicit=True, env={})

    message = assert_safe_message(caught.value, "42", "-1", "-0.5", "raw-secret-key")
    assert "paths.db" in message


def test_path_expansion_for_config_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    path = write_config(
        tmp_path,
        """
[paths]
db = "~/data/tidemark.db"
runtime_dir = "~/run/tidemark"
log_file = "~/logs/tidemark.log"
retention_dir = "~/retained"
""".strip(),
    )

    config = load_config(path, explicit=True, env={})

    assert config.paths.db == home / "data/tidemark.db"
    assert config.paths.runtime_dir == home / "run/tidemark"
    assert config.paths.log_file == home / "logs/tidemark.log"
    assert config.paths.retention_dir == home / "retained"


def test_tidemark_config_env_discovers_config_file(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[search]\ncontext_seconds = 9.5\n")

    config = load_config(env={"TIDEMARK_CONFIG": str(path)})

    assert config.search.context_seconds == 9.5


def test_environment_parsing_and_errors_are_redacted(tmp_path: Path) -> None:
    config = load_config(
        env={
            "TIDEMARK_DB": "~/env.db",
            "TIDEMARK_RUNTIME_DIR": str(tmp_path / "runtime"),
            "TIDEMARK_LOG_FILE": str(tmp_path / "logs" / "tidemark.log"),
            "TIDEMARK_RETENTION_DIR": str(tmp_path / "retained"),
            "TIDEMARK_MONITOR_STREAM_TYPE": "hls",
            "TIDEMARK_MONITOR_TIMEOUT_SECONDS": "3.25",
            "TIDEMARK_INGEST_FINGERPRINT": "yes",
            "TIDEMARK_LOOKUP_TIMEOUT_SECONDS": "7",
            "ACOUSTID_API_KEY": "env-secret-key",
        }
    )

    assert config.paths.db == Path("~/env.db").expanduser()
    assert config.monitor.stream_type == "hls"
    assert config.monitor.timeout_seconds == 3.25
    assert config.ingest.fingerprint is True
    assert config.fingerprint.lookup_timeout_seconds == 7.0
    assert config.fingerprint.api_key == "env-secret-key"

    with pytest.raises(ConfigError) as caught:
        load_config(env={"TIDEMARK_INGEST_FINGERPRINT": "definitely"})
    message = assert_safe_message(caught.value, "definitely")
    assert "TIDEMARK_INGEST_FINGERPRINT" in message
    assert "ingest.fingerprint" in message


def test_cli_overrides_env_over_config_over_defaults(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
[paths]
db = "config.db"

[monitor]
stream_type = "icecast"
timeout_seconds = 12

[search]
context_seconds = 2
""".strip(),
    )
    config = load_config(
        path,
        explicit=True,
        env={"TIDEMARK_DB": "env.db", "TIDEMARK_MONITOR_TIMEOUT_SECONDS": "8"},
    )

    monitor = resolve_monitor_options(config, MonitorOverrides(stream_type="udp"))
    search = resolve_search_options(config, SearchOverrides(context_seconds=11.0))

    assert monitor.db_path == Path("env.db")
    assert monitor.stream_type == "udp"
    assert monitor.timeout_seconds == 8.0
    assert search.db_path == Path("env.db")
    assert search.context_seconds == 11.0


def test_command_resolvers_expose_defaults_and_overrides(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
[paths]
db = "config.db"
retention_dir = "retained"

[ingest]
include_manifest_markers = false
fingerprint = true

[report]
min_score = 0.9
min_count = 3

[fingerprint]
api_key = "config-secret"
lookup_timeout_seconds = 4
""".strip(),
    )
    config = load_config(path, explicit=True, env={})

    ingest = resolve_ingest_options(config, IngestOverrides(include_manifest_markers=True))
    report = resolve_report_options(config)
    clip = resolve_clip_options(config)

    assert ingest.db_path == Path("config.db")
    assert ingest.include_manifest_markers is True
    assert ingest.fingerprint is True
    assert ingest.acoustid_api_key == "config-secret"
    assert ingest.lookup_timeout_seconds == 4.0
    assert report.min_score == 0.9
    assert report.min_count == 3
    assert clip.db_path == Path("config.db")


def test_redaction_masks_api_keys_urls_and_private_paths(tmp_path: Path) -> None:
    raw_path = str(tmp_path / "private" / "tidemark.db")
    assert redact_value("fingerprint.api_key", "secret-value") == "[redacted]"
    assert redact_value("source_url", "https://token@example.com/stream") == "[redacted-url]"
    redacted_path = redact_value("paths.db", raw_path)
    assert redacted_path == "[redacted-path]"
    assert raw_path not in redacted_path


def test_public_config_api_has_no_typer_dependency() -> None:
    import tidemark.config as config_module

    assert "typer" not in config_module.__dict__
