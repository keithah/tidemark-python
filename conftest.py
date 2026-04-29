"""Root conftest: suppress GitHub Actions FORCE_COLOR so Typer/Rich CliRunner
output stays plain-text for assertions that check option names like --db."""
import os

os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
