from __future__ import annotations

import subprocess

import pytest

from parity_support import CommandResult, normalize_marker, normalize_markers, parse_ndjson, run_command, serve_http_fixture


GO_SPLICE_NULL_MARKER = {
    "Type": "SCTE35",
    "Classification": "UNKNOWN",
    "Source": "mpegts",
    "Fields": {"CommandName": "Splice Null"},
}

PYTHON_SPLICE_NULL_MARKER = {
    "Type": "SCTE35",
    "Classification": "UNKNOWN",
    "Source": "mpegts",
    "Tag": None,
    "PTS": None,
    "Segment": None,
    "RawBase64": "/DARAAAAAAAAAP/wAAAAAHpPv/8=",
    "Command": {"name": "Splice Null"},
    "Descriptors": [],
    "Tags": [],
    "Fields": {"CommandName": "Splice Null"},
    "Timestamp": 123.456,
}


def test_normalize_marker_matches_observed_python_and_go_mpegts_shapes() -> None:
    assert normalize_marker(PYTHON_SPLICE_NULL_MARKER) == normalize_marker(GO_SPLICE_NULL_MARKER)
    assert normalize_marker(PYTHON_SPLICE_NULL_MARKER) == GO_SPLICE_NULL_MARKER


def test_normalize_marker_treats_missing_and_null_optionals_equally() -> None:
    with_nulls = {
        "Type": "SCTE35",
        "Classification": "UNKNOWN",
        "Source": "mpegts",
        "Tag": None,
        "PTS": None,
        "Segment": None,
        "Tags": [],
        "Fields": {},
        "Timestamp": 100.0,
    }
    missing = {
        "Type": "SCTE35",
        "Classification": "UNKNOWN",
        "Source": "mpegts",
    }

    assert normalize_marker(with_nulls) == normalize_marker(missing)
    assert normalize_marker(with_nulls) == missing


def test_normalize_marker_preserves_non_empty_fields_tags_and_classification() -> None:
    marker = {
        "Type": "SCTE35",
        "Classification": "AD_BREAK",
        "Source": "hls_manifest",
        "Fields": {"CommandName": "Splice Insert", "BreakDuration": "30.000"},
        "Tags": ["provider:fixture"],
        "Timestamp": 200.0,
    }

    assert normalize_marker(marker) == {
        "Type": "SCTE35",
        "Classification": "AD_BREAK",
        "Source": "hls_manifest",
        "Fields": {"CommandName": "Splice Insert", "BreakDuration": "30.000"},
        "Tags": ["provider:fixture"],
    }


def test_normalize_markers_preserves_order_by_default_and_sorts_only_when_requested() -> None:
    first = {"Type": "ID3", "Classification": "UNKNOWN", "Source": "hls", "Fields": {"ID": "2"}}
    second = {"Type": "SCTE35", "Classification": "UNKNOWN", "Source": "mpegts", "Fields": {"ID": "1"}}

    assert normalize_markers([first, second]) == [first, second]
    assert normalize_markers([first, second], sort=True) == [second, first]


def test_parse_ndjson_reports_malformed_line_number_and_sanitized_label() -> None:
    with pytest.raises(AssertionError) as exc_info:
        parse_ndjson('{"ok": true}\nnot-json\n', label="python http://example.test/live.ts?token=secret")

    message = str(exc_info.value)
    assert "line 2" in message
    assert "python http://example.test/live.ts" in message
    assert "token=secret" not in message


def test_parse_ndjson_empty_stdout_returns_empty_list() -> None:
    assert parse_ndjson("", label="empty") == []


def test_command_result_failure_diagnostics_strip_url_queries() -> None:
    result = CommandResult(
        label="python http://example.test/live.ts?token=secret",
        args=["tidemark", "http://example.test/live.ts?token=secret"],
        returncode=2,
        stdout="",
        stderr="failed for http://example.test/live.ts?token=secret",
    )

    with pytest.raises(AssertionError) as exc_info:
        result.assert_success()

    message = str(exc_info.value)
    assert "http://example.test/live.ts" in message
    assert "token=secret" not in message


def test_run_command_uses_argument_lists_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="{}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_command("fixture", ["tidemark", "monitor", "fixture.ts"])

    assert result.returncode == 0
    assert calls == [
        {
            "args": ["tidemark", "monitor", "fixture.ts"],
            "cwd": None,
            "text": True,
            "capture_output": True,
            "timeout": 30.0,
            "check": False,
        }
    ]
    assert "shell" not in calls[0]


def test_serve_http_fixture_serves_only_registered_paths_and_shuts_down() -> None:
    import urllib.error
    import urllib.request

    with serve_http_fixture({"/fixture.ts": (b"fixture-bytes", "video/MP2T")}) as base_url:
        response = urllib.request.urlopen(f"{base_url}/fixture.ts?token=secret", timeout=5)
        try:
            assert response.status == 200
            assert response.headers["Content-Type"] == "video/MP2T"
            assert response.read() == b"fixture-bytes"
        finally:
            response.close()

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}/missing.ts?token=secret", timeout=5)
        assert exc_info.value.code == 404

    with pytest.raises(OSError):
        urllib.request.urlopen(f"{base_url}/fixture.ts", timeout=0.1)
