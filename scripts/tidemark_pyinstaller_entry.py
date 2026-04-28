"""PyInstaller entry point for the standalone tidemark executable."""

from __future__ import annotations

from tidemark.cli.main import app


def main() -> None:
    """Run the Typer app through a source-controlled freeze entrypoint."""
    app()


if __name__ == "__main__":
    main()
