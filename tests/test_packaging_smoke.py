from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scte35_splice_null.ts"
EXPECTED_HELP_COMMANDS = ("monitor", "ingest", "status", "search", "report", "clip")
EXPECTED_MARKER_KEYS = [
    "Type",
    "Classification",
    "Source",
    "Tag",
    "PTS",
    "Segment",
    "RawBase64",
    "Command",
    "Descriptors",
    "Tags",
    "Fields",
    "Timestamp",
]
ENV_REMOVE_EXACT = {
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
    "ACOUSTID_API_KEY",
    "TIDEMARK_CONFIG",
    "TIDEMARK_DB",
    "TIDEMARK_RUNTIME_DIR",
}
ENV_REMOVE_TOKEN = ("ACOUSTID", "PYTHON", "VENV")


@pytest.fixture(scope="session")
def packaged_cli() -> Path:
    raw = os.environ.get("TIDEMARK_PACKAGED_CLI")
    if not raw:
        pytest.skip("set TIDEMARK_PACKAGED_CLI to run packaged CLI smoke tests")
    cli = Path(raw).expanduser().resolve()
    if not cli.exists():
        pytest.fail(f"TIDEMARK_PACKAGED_CLI points to a missing path: {cli}")
    if not cli.is_file():
        pytest.fail(f"TIDEMARK_PACKAGED_CLI is not a file: {cli}")
    if not os.access(cli, os.X_OK):
        pytest.fail(f"TIDEMARK_PACKAGED_CLI is not executable: {cli}")
    return cli


@pytest.fixture()
def clean_env() -> dict[str, str]:
    env: dict[str, str] = {}
    root_text = str(ROOT)
    for key, value in os.environ.items():
        if key in ENV_REMOVE_EXACT:
            continue
        if any(token in key.upper() for token in ENV_REMOVE_TOKEN):
            continue
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR"}:
            env[key] = value
            continue
        if isinstance(value, str) and root_text in value:
            continue
        env[key] = value
    return env


def run_packaged(
    packaged_cli: Path,
    clean_env: dict[str, str],
    *args: object,
    cwd: Path,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    command = [str(packaged_cli), *[str(arg) for arg in args]]
    return subprocess.run(
        command,
        cwd=cwd,
        env=clean_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def parse_ndjson(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    assert "Traceback" not in result.stdout, _format_result(result)
    rows: list[dict[str, object]] = []
    for index, line in enumerate(result.stdout.splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"invalid NDJSON line {index}: {exc}\n{_format_result(result)}")
        assert isinstance(payload, dict), _format_result(result)
        rows.append(payload)
    return rows


def comparable_marker(marker: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in marker.items() if key != "Timestamp"}


def copy_monitor_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "fixture.ts"
    shutil.copy2(FIXTURE, destination)
    assert destination.exists(), f"expected copied fixture at {destination}"
    return destination


def assert_no_traceback_or_private_path(result: subprocess.CompletedProcess[str], *private_paths: Path) -> None:
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, _format_result(result)
    for private_path in private_paths:
        private_text = str(private_path)
        assert private_text not in combined, _format_result(result)


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command={result.args!r}\n"
        f"cwd={result.cwd if hasattr(result, 'cwd') else '<captured by subprocess>'}\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_packaged_executable_exists_and_looks_like_linux_binary(packaged_cli: Path) -> None:
    assert packaged_cli.is_absolute()
    assert packaged_cli.exists()
    assert os.access(packaged_cli, os.X_OK)
    if os.name == "posix" and "linux" in os.sys.platform:
        assert packaged_cli.read_bytes()[:4] == b"\x7fELF"


def test_packaged_help_lists_core_commands(
    packaged_cli: Path,
    clean_env: dict[str, str],
    tmp_path: Path,
) -> None:
    result = run_packaged(packaged_cli, clean_env, "--help", cwd=tmp_path)

    assert result.returncode == 0, _format_result(result)
    assert result.stderr == "", _format_result(result)
    assert_no_traceback_or_private_path(result, tmp_path)
    for command_name in EXPECTED_HELP_COMMANDS:
        assert command_name in result.stdout, _format_result(result)


def test_packaged_status_missing_runtime_dir_reports_no_runs_without_creating_sqlite(
    packaged_cli: Path,
    clean_env: dict[str, str],
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    expected_db = tmp_path / "tidemark.db"

    result = run_packaged(packaged_cli, clean_env, "status", "--runtime-dir", runtime_dir, cwd=tmp_path)

    assert result.returncode == 0, _format_result(result)
    assert result.stderr == "", _format_result(result)
    assert "No runtime health records found; tidemark is not running." in result.stdout, _format_result(result)
    assert "Runtime directory: runtime" in result.stdout, _format_result(result)
    assert not runtime_dir.exists()
    assert not expected_db.exists()
    assert_no_traceback_or_private_path(result, tmp_path, runtime_dir)


def test_packaged_monitor_and_root_alias_emit_matching_marker_shape(
    packaged_cli: Path,
    clean_env: dict[str, str],
    tmp_path: Path,
) -> None:
    fixture_copy = copy_monitor_fixture(tmp_path)

    canonical = run_packaged(
        packaged_cli,
        clean_env,
        "monitor",
        fixture_copy.name,
        "--stream-type",
        "mpegts",
        "--json",
        "--timeout",
        "1",
        cwd=tmp_path,
    )
    alias = run_packaged(
        packaged_cli,
        clean_env,
        fixture_copy.name,
        "--stream-type",
        "mpegts",
        "--json",
        "--timeout",
        "1",
        cwd=tmp_path,
    )

    assert canonical.returncode == 0, _format_result(canonical)
    assert alias.returncode == 0, _format_result(alias)
    assert canonical.stderr.startswith("[tidemark] completed:"), _format_result(canonical)
    assert alias.stderr.startswith("[tidemark] completed:"), _format_result(alias)
    assert_no_traceback_or_private_path(canonical, tmp_path, fixture_copy)
    assert_no_traceback_or_private_path(alias, tmp_path, fixture_copy)

    canonical_markers = parse_ndjson(canonical)
    alias_markers = parse_ndjson(alias)
    assert canonical_markers, _format_result(canonical)
    assert len(alias_markers) == len(canonical_markers), (
        f"canonical_count={len(canonical_markers)} alias_count={len(alias_markers)}\n"
        f"canonical:\n{_format_result(canonical)}\n"
        f"alias:\n{_format_result(alias)}"
    )

    first = canonical_markers[0]
    assert list(first) == EXPECTED_MARKER_KEYS, _format_result(canonical)
    assert first["Type"] == "SCTE35", _format_result(canonical)
    assert first["Source"] == "mpegts", _format_result(canonical)
    assert first["Classification"] == "UNKNOWN", _format_result(canonical)
    assert first["Fields"] == {"CommandName": "Splice Null"}, _format_result(canonical)
    assert [comparable_marker(marker) for marker in alias_markers] == [
        comparable_marker(marker) for marker in canonical_markers
    ]
