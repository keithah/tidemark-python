"""User-facing tidemark command line entry point."""

from __future__ import annotations

import click
import typer
from typer.core import TyperGroup

from tidemark.cli.cmd_monitor import monitor
from tidemark.cli.cmd_search import search


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


app = typer.Typer(
    add_completion=False,
    cls=RootAliasGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def root() -> None:
    """Detect ad markers in HLS, ICY, MPEG-TS, and UDP streams."""


app.command(name="monitor")(monitor)
app.command(name="search")(search)

__all__ = ["RootAliasGroup", "app", "root"]
