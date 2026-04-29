"""tidemark doctor — preflight environment check."""

from __future__ import annotations

import typer

from tidemark.runtime.doctor import run_checks


def doctor() -> None:
    """Check that the tidemark runtime environment is correctly configured."""
    results = run_checks()
    any_failed = False
    for result in results:
        mark = "[ok]" if result.ok else "[!!]"
        typer.echo(f"{mark} {result.label}: {result.detail}")
        if not result.ok:
            any_failed = True

    if any_failed:
        raise typer.Exit(code=1)
