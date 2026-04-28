from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from parity_support import (
    build_go_cli,
    normalize_markers,
    parse_ndjson,
    python_cli_path,
    run_command,
    serve_http_fixture,
)
from tests.test_icy_ingest import build_icy_stream
from tests.test_id3_markers import build_id3_tag

FIXTURE_PATH = Path("tests/fixtures/scte35_splice_null.ts")
MPEGTS_ROUTE = "/scte35_splice_null.ts"
TOKEN_VALUE = "secret-token-value"
RAW_SPLICE_NULL_BASE64 = "/DARAAAAAAAAAP/wAAAAAHpPv/8="
HLS_PLAYLIST_ROUTE = "/hls/playlist.m3u8"
HLS_SEGMENT_ROUTE = "/hls/segment0.ts"
ICY_ROUTE = "/icy"


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


@pytest.fixture(scope="session")
def hls_segment_bytes() -> bytes:
    return b"prefix" + build_id3_tag(title="Segment marker", private_data=b"private-payload")


@pytest.fixture(scope="session")
def hls_playlist_bytes() -> bytes:
    return (
        "#EXTM3U\n"
        "#EXT-X-TARGETDURATION:1\n"
        "#EXT-X-VERSION:3\n"
        "#EXT-X-MEDIA-SEQUENCE:0\n"
        f"#EXT-X-SCTE35:{RAW_SPLICE_NULL_BASE64}\n"
        "#EXTINF:1.0,\n"
        f"{HLS_SEGMENT_ROUTE.rsplit('/', 1)[-1]}\n"
        "#EXT-X-ENDLIST\n"
    ).encode("utf-8")


@pytest.fixture()
def hls_url(hls_playlist_bytes: bytes, hls_segment_bytes: bytes):
    with serve_http_fixture(
        {
            HLS_PLAYLIST_ROUTE: (hls_playlist_bytes, "application/vnd.apple.mpegurl"),
            HLS_SEGMENT_ROUTE: (hls_segment_bytes, "video/MP2T"),
        }
    ) as base_url:
        yield f"{base_url}{HLS_PLAYLIST_ROUTE}"


@pytest.fixture(scope="session")
def icy_stream_bytes() -> bytes:
    return build_icy_stream(
        16,
        [b"StreamTitle='Promo Spot';StreamUrl='https://station.example/ad';"],
    )


@pytest.fixture()
def icy_url(icy_stream_bytes: bytes):
    with serve_http_fixture(
        {
            ICY_ROUTE: (
                icy_stream_bytes,
                "audio/mpeg",
                {"icy-metaint": "16"},
            )
        }
    ) as base_url:
        yield f"{base_url}{ICY_ROUTE}"


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


def test_hls_http_fixture_matches_go_common_marker_fields(
    python_cli: Path,
    go_binary: Path,
    hls_url: str,
) -> None:
    python_result = run_command(
        "python HLS SCTE-35 and ID3 fixture",
        [python_cli, "monitor", hls_url, "--stream-type", "hls", "--json", "--timeout", "3"],
        timeout=8,
    )
    go_result = run_command(
        "go HLS SCTE-35 and ID3 fixture",
        [go_binary, "--json", "--timeout", "3", hls_url],
        timeout=8,
    )

    assert_cli_health(python_result)
    assert_cli_health(go_result)
    python_markers = parse_ndjson(python_result.stdout, label="python HLS SCTE-35 and ID3 fixture")
    go_markers = parse_ndjson(go_result.stdout, label="go HLS SCTE-35 and ID3 fixture")

    assert python_markers, "python canonical emitted no HLS markers"
    assert go_markers, "go reference emitted no HLS markers"
    assert _marker_projection(python_markers) <= _marker_projection(go_markers)
    assert {marker["Source"] for marker in python_markers} == {"hls_manifest", "hls_segment"}
    assert {marker["Type"] for marker in python_markers} == {"SCTE35", "ID3"}
    assert any(marker["Classification"] == "AD_START" for marker in python_markers if marker["Type"] == "ID3")

    python_scte35 = [marker for marker in python_markers if marker["Type"] == "SCTE35"]
    go_scte35 = [marker for marker in go_markers if marker["Type"] == "SCTE35"]
    assert _drop_keys(normalize_markers(python_scte35), {"Segment"}) == _drop_keys(
        normalize_markers(go_scte35), {"Segment"}
    )


def test_icy_http_fixture_matches_go_after_normalization(
    python_cli: Path,
    go_binary: Path,
    icy_url: str,
) -> None:
    python_result = run_command(
        "python ICY fixture",
        [python_cli, "monitor", icy_url, "--stream-type", "icy", "--json", "--timeout", "2"],
        timeout=8,
    )
    go_result = run_command(
        "go ICY fixture",
        [go_binary, "--json", "--timeout", "2", icy_url],
        timeout=8,
    )

    assert_cli_health(python_result)
    assert_cli_health(go_result)
    python_markers = parse_ndjson(python_result.stdout, label="python ICY fixture")
    go_markers = parse_ndjson(go_result.stdout, label="go ICY fixture")

    assert python_markers, "python canonical emitted no ICY markers"
    assert go_markers, "go reference emitted no ICY markers"
    assert normalize_markers(python_markers) == normalize_markers(go_markers)
    assert python_markers[0]["Source"] == "icy_stream"
    assert python_markers[0]["Type"] == "ICY"
    assert python_markers[0]["Classification"] == "AD_START"


def _marker_projection(markers: list[dict[str, Any]]) -> set[tuple[Any, Any, Any]]:
    return {
        (
            marker.get("Type"),
            marker.get("Classification"),
            marker.get("Source"),
        )
        for marker in markers
    }


def _drop_keys(markers: list[dict[str, Any]], keys: set[str]) -> list[dict[str, Any]]:
    return [{key: value for key, value in marker.items() if key not in keys} for marker in markers]


def assert_cli_health(result) -> None:
    result.assert_success()
    assert "Traceback" not in result.stderr
    assert TOKEN_VALUE not in result.stderr
    assert RAW_SPLICE_NULL_BASE64 not in result.stderr
