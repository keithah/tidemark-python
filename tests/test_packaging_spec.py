from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
ENTRY = ROOT / "scripts" / "tidemark_pyinstaller_entry.py"
SPEC = ROOT / "tidemark.spec"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def test_packaging_dependencies_are_declared_for_runtime_and_dev_extras():
    config = tomllib.loads(PYPROJECT.read_text())

    dependencies = config["project"]["dependencies"]
    dev_dependencies = config["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("click>=8") for dep in dependencies)
    assert any(dep.startswith("threefive>=3.0,<3.0.78") for dep in dependencies)
    assert any(dep.startswith("typer>=0.12,<0.26") for dep in dependencies)
    assert any(dep.startswith("pyinstaller>=6") for dep in dev_dependencies)
    assert any(dep.startswith("staticx>=0.14") for dep in dev_dependencies)
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
    assert "pathex=[\"src\", threefive_source_root]" in source or "pathex=['src', threefive_source_root]" in source
    assert "collect_submodules(\"tidemark\")" in source or "collect_submodules('tidemark')" in source
    assert "threefive" in source
    assert "collect_all(\"av\")" in source or "collect_all('av')" in source
    assert "name=\"tidemark\"" in source or "name='tidemark'" in source
    assert "console=True" in source
    assert "onefile" not in source.lower(), "onefile should be expressed by a single-file EXE, not comments only"


def test_release_workflow_wraps_linux_artifact_as_static_executable():
    source = RELEASE_WORKFLOW.read_text()

    assert "staticx" in source
    assert "ubuntu-latest" in source
    assert "patchelf" in source
    assert "squashfs-tools" in source
    assert "dist/tidemark-linux-x86_64" in source
    assert "ldd" in source
    assert "ldd \"$bin\" > /tmp/tidemark-linux-ldd.txt 2>&1 || true" in source
    assert "not a dynamic executable" in source
