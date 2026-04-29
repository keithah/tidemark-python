"""User-facing tidemark command line entry point."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Annotated

import click
import typer
from typer.core import TyperGroup

from tidemark import __version__ as _package_version
from tidemark.cli.cmd_clip import clip
from tidemark.cli.cmd_ingest import ingest
from tidemark.cli.cmd_monitor import monitor
from tidemark.cli.cmd_report import report
from tidemark.cli.cmd_search import search
from tidemark.cli.cmd_status import status


class RootAliasGroup(TyperGroup):
    """Route `tidemark <url> ...` through the canonical `monitor` command."""

    def resolve_command(self, ctx: click.Context, args: list[str]):  # type: ignore[override]
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if not args:
                raise
            monitor_command = self.get_command(ctx, "monitor")
            if monitor_command is None:
                raise
            url = args[0]
            remaining = args[1:]
            return "monitor", monitor_command, [url, *remaining]


def _cli_version() -> str:
    try:
        return _pkg_version("tidemark")
    except PackageNotFoundError:
        return _package_version


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tidemark {_cli_version()}")
        raise typer.Exit()


app = typer.Typer(
    add_completion=False,
    cls=RootAliasGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """Detect ad markers in HLS, ICY, MPEG-TS, and UDP streams."""


app.command(name="ingest")(ingest)
app.command(name="monitor")(monitor)
app.command(name="status")(status)
app.add_typer(report, name="report")
app.command(name="search")(search)
app.command(name="clip")(clip)

__all__ = ["RootAliasGroup", "app", "root"]
