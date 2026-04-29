"""Shared pytest fixtures and setup for the tidemark test suite."""
import os

import pytest


@pytest.fixture(autouse=True)
def _disable_force_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove FORCE_COLOR so Typer/Rich CliRunner output stays plain-text.

    GitHub Actions sets FORCE_COLOR=1 which causes Rich to insert ANSI
    escape codes inside option names (splitting --db into -<ESC>-db),
    breaking assertions that check for exact option name substrings.
    """
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
