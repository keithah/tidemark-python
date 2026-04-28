from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
ENTRY = ROOT / "scripts" / "tidemark_pyinstaller_entry.py"
SPEC = ROOT / "tidemark.spec"


def test_packaging_dependencies_are_declared_for_runtime_and_dev_extras():
    config = tomllib.loads(PYPROJECT.read_text())

    dependencies = config["project"]["dependencies"]
    dev_dependencies = config["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("threefive>=3.0") for dep in dependencies)
    assert any(dep.startswith("pyinstaller>=6") for dep in dev_dependencies)
    assert config["project"]["optional-dependencies"]["fingerprint"] == ["pyacoustid>=1.3"]


def test_pyinstaller_entry_script_delegates_to_typer_app_without_venv_paths():
    source = ENTRY.read_text()
    tree = ast.parse(source)

    assert "tidemark.cli.main" in source
    assert "app" in source
    assert ".venv" not in source
    assert "tests" not in source
    assert any(isinstance(node, ast.If) for node in tree.body), "entrypoint must have __main__ guard"


def test_pyinstaller_spec_freezes_onefile_console_binary_and_collects_runtime_data():
    source = SPEC.read_text()

    assert "scripts/tidemark_pyinstaller_entry.py" in source
    assert "pathex=[\"src\"]" in source or "pathex=['src']" in source
    assert "collect_submodules(\"tidemark\")" in source or "collect_submodules('tidemark')" in source
    assert "collect_all(\"imageio_ffmpeg\")" in source or "collect_all('imageio_ffmpeg')" in source
    assert "name=\"tidemark\"" in source or "name='tidemark'" in source
    assert "console=True" in source
    assert "onefile" not in source.lower(), "onefile should be expressed by a single-file EXE, not comments only"
