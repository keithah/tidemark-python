"""Runtime health status command for the tidemark Typer CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tidemark.config import ConfigError, default_runtime_dir, load_config
from tidemark.runtime.health import format_status_report, read_status_entries

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="TOML config file to load for command defaults."),
]
RuntimeDirOption = Annotated[
    Path | None,
    typer.Option("--runtime-dir", help="Runtime directory containing tidemark run health files."),
]


def run_status_command(
    *,
    config_path: Path | None = None,
    runtime_dir: Path | None = None,
) -> None:
    """Load config once, read runtime health records, and print a side-effect-safe report."""
    try:
        config = load_config(config_path, explicit=config_path is not None)
    except ConfigError as exc:
        _fatal(str(exc))

    resolved_runtime_dir = Path(runtime_dir or config.paths.runtime_dir or default_runtime_dir()).expanduser()
    entries, diagnostics = read_status_entries(resolved_runtime_dir)
    typer.echo(format_status_report(entries, diagnostics, runtime_dir=resolved_runtime_dir))


def status(
    config_path: ConfigOption = None,
    runtime_dir: RuntimeDirOption = None,
) -> None:
    """Show runtime health for active and recent tidemark runs."""
    run_status_command(config_path=config_path, runtime_dir=runtime_dir)


def _fatal(message: str) -> None:
    typer.echo(f"[tidemark] error: {message}", err=True)
    raise typer.Exit(code=1)


__all__ = ["run_status_command", "status"]
