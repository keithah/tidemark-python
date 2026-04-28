from __future__ import annotations

from pathlib import Path

import pytest

from parity_support import build_go_cli, normalize_markers, parse_ndjson, python_cli_path, run_command, serve_http_fixture

FIXTURE_PATH = Path("tests/fixtures/scte35_splice_null.ts")
MPEGTS_ROUTE = "/scte35_splice_null.ts"
TOKEN_VALUE = "secret-token-value"
RAW_SPLICE_NULL_BASE64 = "/DARAAAAAAAAAP/wAAAAAHpPv/8="


@pytest.fixture(scope="session")
def python_cli() -> Path:
    return python_cli_path()


@pytest.fixture(scope="session")
def go_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_go_cli(tmp_path_factory.mktemp("go-cli"))


@pytest.fixture(scope="session")
def mpegts_fixture_bytes() -> bytes:
    assert FIXTURE_PATH.exists(), f"expected tracked MPEGTS fixture at {FIXTURE_PATH}"
    return FIXTURE_PATH.read_bytes()


@pytest.fixture()
def mpegts_url(mpegts_fixture_bytes: bytes):
    with serve_http_fixture({MPEGTS_ROUTE: (mpegts_fixture_bytes, "video/MP2T")}) as base_url:
        yield f"{base_url}{MPEGTS_ROUTE}"


def test_mpegts_http_fixture_matches_go_after_normalization(
    python_cli: Path,
    go_binary: Path,
    mpegts_url: str,
) -> None:
    python_result = run_command(
        "python canonical MPEGTS fixture",
        [python_cli, "monitor", mpegts_url, "--stream-type", "mpegts", "--json", "--timeout", "1"],
    )
    go_result = run_command(
        "go MPEGTS fixture",
        [go_binary, "--json", "--timeout", "1", mpegts_url],
    )

    assert_cli_health(python_result)
    assert_cli_health(go_result)
    python_markers = parse_ndjson(python_result.stdout, label="python canonical MPEGTS fixture")
    go_markers = parse_ndjson(go_result.stdout, label="go MPEGTS fixture")

    assert python_markers, "python canonical emitted no MPEGTS markers"
    assert go_markers, "go reference emitted no MPEGTS markers"
    assert normalize_markers(python_markers) == normalize_markers(go_markers)


def test_root_alias_mpegts_fixture_matches_go_after_normalization(
    python_cli: Path,
    go_binary: Path,
    mpegts_url: str,
) -> None:
    canonical_result = run_command(
        "python canonical MPEGTS fixture",
        [python_cli, "monitor", mpegts_url, "--stream-type", "mpegts", "--json", "--timeout", "1"],
    )
    alias_result = run_command(
        "python root alias MPEGTS fixture",
        [python_cli, mpegts_url, "--stream-type", "mpegts", "--json", "--timeout", "1"],
    )
    go_result = run_command(
        "go MPEGTS fixture",
        [go_binary, "--json", "--timeout", "1", mpegts_url],
    )

    assert_cli_health(canonical_result)
    assert_cli_health(alias_result)
    assert_cli_health(go_result)
    canonical_markers = parse_ndjson(canonical_result.stdout, label="python canonical MPEGTS fixture")
    alias_markers = parse_ndjson(alias_result.stdout, label="python root alias MPEGTS fixture")
    go_markers = parse_ndjson(go_result.stdout, label="go MPEGTS fixture")

    assert alias_markers, "python root alias emitted no MPEGTS markers"
    assert canonical_markers, "python canonical emitted no MPEGTS markers"
    assert go_markers, "go reference emitted no MPEGTS markers"
    assert normalize_markers(alias_markers) == normalize_markers(go_markers)
    assert normalize_markers(alias_markers) == normalize_markers(canonical_markers)


def test_mpegts_filter_and_json_out_match_go_supported_stdout(
    python_cli: Path,
    go_binary: Path,
    mpegts_url: str,
    tmp_path: Path,
) -> None:
    python_json_out = tmp_path / "python.ndjson"
    go_json_out = tmp_path / "go.ndjson"
    python_result = run_command(
        "python filtered MPEGTS fixture",
        [
            python_cli,
            "monitor",
            mpegts_url,
            "--stream-type",
            "mpegts",
            "--filter",
            "scte35",
            "--json",
            "--json-out",
            python_json_out,
            "--timeout",
            "1",
        ],
    )
    go_result = run_command(
        "go filtered MPEGTS fixture",
        [go_binary, "--filter", "scte35", "--json", "--json-out", go_json_out, "--timeout", "1", mpegts_url],
    )

    assert_cli_health(python_result)
    assert_cli_health(go_result)
    python_markers = parse_ndjson(python_result.stdout, label="python filtered MPEGTS fixture")
    go_markers = parse_ndjson(go_result.stdout, label="go filtered MPEGTS fixture")

    assert python_markers, "python filtered run emitted no MPEGTS markers"
    assert go_markers, "go filtered run emitted no MPEGTS markers"
    assert normalize_markers(python_markers) == normalize_markers(go_markers)
    assert python_json_out.read_text() == python_result.stdout
    assert go_json_out.read_text() == go_result.stdout


def assert_cli_health(result) -> None:
    result.assert_success()
    assert "Traceback" not in result.stderr
    assert TOKEN_VALUE not in result.stderr
    assert RAW_SPLICE_NULL_BASE64 not in result.stderr
