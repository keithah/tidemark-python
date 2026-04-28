from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tidemark.markers import AD_END, AD_START, UNKNOWN, AdMarker
from tidemark.monitor import MonitorOptions, MonitorResult, run_monitor
from tidemark.monitor_sources import MonitorSourceError, StreamType
from tidemark.runtime.retry import RetryPolicy
from tidemark.store import migrate


def marker(
    marker_type: str = "HLS",
    *,
    tag: str | None = "#EXT-X-CUE-OUT",
    source: str = "fixture?token=secret",
    classification: str = UNKNOWN,
    timestamp: float = 1.0,
    fields: dict[str, object] | None = None,
) -> AdMarker:
    return AdMarker(
        type=marker_type,
        classification=classification,
        source=source,
        tag=tag,
        timestamp=timestamp,
        fields=fields or {},
        raw_base64="secret-payload",
    )


def run_with(markers: object, options: MonitorOptions | None = None) -> tuple[MonitorResult, io.StringIO, io.StringIO]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = run_monitor(markers, options=options or MonitorOptions(source_url="fixture://stream?token=secret"), stdout=stdout, stderr=stderr)
    return result, stdout, stderr


def ndjson_lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_run_monitor_classifies_filters_and_writes_go_compatible_stdout() -> None:
    markers = [
        marker(tag="#EXT-X-CUE-OUT", timestamp=1.0),
        marker(tag="#EXT-X-CUE-IN", timestamp=2.0),
        marker(marker_type="OTHER", tag=None, timestamp=3.0),
    ]

    result, stdout, stderr = run_with(markers, MonitorOptions(source_url="https://media.example/live.m3u8?token=secret", marker_filter=AD_START))

    assert result == MonitorResult(reason="eof", markers_seen=3, markers_emitted=1, markers_filtered=2)
    assert [item.classification for item in markers] == [AD_START, AD_END, UNKNOWN]
    assert ndjson_lines(stdout) == [markers[0].to_dict()]
    assert "raw_base64" not in stdout.getvalue()
    assert "RawBase64" in stdout.getvalue()
    assert "[tidemark] completed: reason=eof markers=3 emitted=1 filtered=2" in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()


def test_run_monitor_progress_callback_observes_running_counters_and_eof_terminal() -> None:
    events: list[tuple[str, str | None, dict[str, int], str | None]] = []
    first = marker(tag="#EXT-X-CUE-OUT")
    second = marker(marker_type="OTHER", tag=None)

    def record(progress) -> None:
        events.append((progress.phase, progress.reason, dict(progress.counters), progress.error))

    result, stdout, stderr = run_with(
        [first, second],
        MonitorOptions(source_url="https://media.example/live.m3u8?token=secret", marker_filter=AD_START, progress_callback=record),
    )

    assert result.reason == "eof"
    assert ndjson_lines(stdout) == [first.to_dict()]
    assert "reason=eof markers=2 emitted=1 filtered=1" in stderr.getvalue()
    assert events == [
        ("running", None, {"markers_seen": 0, "markers_emitted": 0, "markers_filtered": 0, "sink_warnings": 0}, None),
        ("running", None, {"markers_seen": 1, "markers_emitted": 1, "markers_filtered": 0, "sink_warnings": 0}, None),
        ("running", None, {"markers_seen": 2, "markers_emitted": 1, "markers_filtered": 1, "sink_warnings": 0}, None),
        ("completed", "eof", {"markers_seen": 2, "markers_emitted": 1, "markers_filtered": 1, "sink_warnings": 0}, None),
    ]


def test_run_monitor_retries_transient_setup_failure_and_keeps_stdout_marker_only() -> None:
    attempts = 0
    sleeps: list[float] = []
    events: list[object] = []
    first = marker(tag="#EXT-X-CUE-OUT")

    def source_factory():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MonitorSourceError(
                "hls source setup failed token=secret https://media.example/live.m3u8?token=secret",
                stream_type=StreamType.HLS,
                phase="setup",
            )
        return iter([first])

    result, stdout, stderr = run_with(
        source_factory,
        MonitorOptions(
            source_url="https://media.example/live.m3u8?token=secret",
            retry_policy=RetryPolicy(max_attempts=1, initial_backoff_seconds=0.25, max_backoff_seconds=1.0),
            retry_sleep=sleeps.append,
            progress_callback=events.append,
            clock=lambda: 0.0,
        ),
    )

    assert result == MonitorResult(reason="eof", markers_seen=1, markers_emitted=1, markers_filtered=0)
    assert attempts == 2
    assert sleeps == [0.25]
    assert stdout.getvalue().splitlines() == [first.to_json()]
    assert "tidemark" not in stdout.getvalue()
    assert "token=secret" not in stderr.getvalue()
    retry_events = [event for event in events if event.phase == "retrying"]
    assert len(retry_events) == 1
    assert retry_events[0].retry_attempt == 1
    assert retry_events[0].delay_seconds == 0.25
    assert retry_events[0].next_retry_at == datetime.fromtimestamp(0.25, timezone.utc)
    assert retry_events[0].error == "hls source setup failed token=[redacted] media.example/live.m3u8"
    assert retry_events[0].counters == {
        "markers_seen": 0,
        "markers_emitted": 0,
        "markers_filtered": 0,
        "sink_warnings": 0,
    }


def test_run_monitor_retries_transient_iterator_failure_and_preserves_counters() -> None:
    attempts = 0
    sleeps: list[float] = []
    events: list[object] = []
    emitted = marker(tag="#EXT-X-CUE-OUT", timestamp=1.0)
    filtered = marker(marker_type="OTHER", tag=None, timestamp=2.0)
    recovered = marker(tag="#EXT-X-CUE-IN", timestamp=3.0)

    def source_factory():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield emitted
            yield filtered
            raise MonitorSourceError(
                "hls source iteration failed https://media.example/live.m3u8?token=secret",
                stream_type=StreamType.HLS,
                phase="iteration",
            )
        yield recovered

    result, stdout, _stderr = run_with(
        source_factory,
        MonitorOptions(
            source_url="https://media.example/live.m3u8?token=secret",
            marker_filter="ad",
            retry_policy=RetryPolicy(max_attempts=1, initial_backoff_seconds=0.0, max_backoff_seconds=1.0),
            retry_sleep=sleeps.append,
            progress_callback=events.append,
            clock=lambda: 0.0,
        ),
    )

    assert result == MonitorResult(reason="eof", markers_seen=3, markers_emitted=2, markers_filtered=1)
    assert attempts == 2
    assert sleeps == [0.0]
    assert stdout.getvalue().splitlines() == [emitted.to_json(), recovered.to_json()]
    retry_event = next(event for event in events if event.phase == "retrying")
    assert retry_event.counters == {
        "markers_seen": 2,
        "markers_emitted": 1,
        "markers_filtered": 1,
        "sink_warnings": 0,
    }


def test_run_monitor_exhausts_transient_retries_with_redacted_error_snapshot() -> None:
    attempts = 0
    events: list[object] = []

    def source_factory():
        nonlocal attempts
        attempts += 1
        raise MonitorSourceError(
            "icy source setup failed https://media.example/live?token=secret raw_base64=abc",
            stream_type=StreamType.ICY,
            phase="setup",
        )

    result, stdout, stderr = run_with(
        source_factory,
        MonitorOptions(
            source_url="https://media.example/live?token=secret",
            retry_policy=RetryPolicy(max_attempts=1, initial_backoff_seconds=0.0, max_backoff_seconds=1.0),
            retry_sleep=lambda delay: None,
            progress_callback=events.append,
            clock=lambda: 0.0,
        ),
    )

    assert result.reason == "error"
    assert result.error == "icy source setup failed media.example/live raw_base64=[redacted]"
    assert attempts == 2
    assert stdout.getvalue() == ""
    assert "icy source setup failed" in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()
    assert "raw_base64=abc" not in stderr.getvalue()
    assert [event.phase for event in events].count("retrying") == 1
    terminal = events[-1]
    assert terminal.phase == "error"
    assert terminal.reason == "error"
    assert terminal.error == "icy source setup failed media.example/live raw_base64=[redacted]"


def test_run_monitor_does_not_retry_deterministic_monitor_errors() -> None:
    retry_policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0, max_backoff_seconds=1.0)
    sleeps: list[float] = []

    invalid_filter, invalid_stdout, _ = run_with([marker()], MonitorOptions(marker_filter="bogus", retry_policy=retry_policy, retry_sleep=sleeps.append))
    non_marker, non_marker_stdout, _ = run_with([object()], MonitorOptions(retry_policy=retry_policy, retry_sleep=sleeps.append))

    class BrokenMarker(AdMarker):
        def to_json(self) -> str:
            raise TypeError("bad marker token=secret")

    broken_marker = BrokenMarker(type="ID3", classification=UNKNOWN, source="fixture", tag=None, timestamp=1.0)
    serialization, serialization_stdout, _ = run_with([broken_marker], MonitorOptions(retry_policy=retry_policy, retry_sleep=sleeps.append))

    assert invalid_filter.error == "invalid marker filter"
    assert invalid_stdout.getvalue() == ""
    assert non_marker.error == "marker source yielded non-AdMarker value"
    assert non_marker_stdout.getvalue() == ""
    assert serialization.error == "marker serialization failed"
    assert serialization_stdout.getvalue() == ""
    assert sleeps == []


def test_run_monitor_retry_disabled_preserves_iterator_failure_behavior() -> None:
    attempts = 0

    def source_factory():
        nonlocal attempts
        attempts += 1
        raise MonitorSourceError("hls source setup failed", stream_type=StreamType.HLS, phase="setup")

    result, stdout, stderr = run_with(source_factory, MonitorOptions(retry_policy=RetryPolicy(max_attempts=0)))

    assert result.reason == "error"
    assert result.error == "hls source setup failed"
    assert attempts == 1
    assert stdout.getvalue() == ""
    assert "hls source setup failed" in stderr.getvalue()


def test_run_monitor_timeout_bounds_retry_sleep() -> None:
    times = iter([0.0, 0.0, 1.0])
    sleeps: list[float] = []

    def source_factory():
        raise MonitorSourceError("hls source setup failed", stream_type=StreamType.HLS, phase="setup")

    result, stdout, stderr = run_with(
        source_factory,
        MonitorOptions(
            retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=2.0, max_backoff_seconds=10.0),
            retry_sleep=sleeps.append,
            timeout=1.0,
            clock=lambda: next(times),
        ),
    )

    assert result.reason == "timeout"
    assert result.error is None
    assert sleeps == [1.0]
    assert stdout.getvalue() == ""
    assert "reason=timeout" in stderr.getvalue()


def test_run_monitor_progress_callback_failures_do_not_change_output_or_result() -> None:
    calls = 0
    first = marker(tag="#EXT-X-CUE-OUT")

    def broken_callback(progress) -> None:  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise RuntimeError("health write failed token=secret")

    result, stdout, stderr = run_with(
        [first],
        MonitorOptions(source_url="https://media.example/live.m3u8?token=secret", progress_callback=broken_callback),
    )

    assert result == MonitorResult(reason="eof", markers_seen=1, markers_emitted=1, markers_filtered=0)
    assert ndjson_lines(stdout) == [first.to_dict()]
    assert stderr.getvalue() == "[tidemark] completed: reason=eof markers=1 emitted=1 filtered=0\n"
    assert calls >= 2


def test_run_monitor_accepts_callable_marker_source_and_uses_one_classifier_per_run() -> None:
    icy_markers = [
        marker("ICY", tag=None, fields={"StreamTitle": "Morning Show"}),
        marker("ICY", tag=None, fields={"StreamTitle": "Station Promo"}),
        marker("ICY", tag=None, fields={"StreamTitle": "Morning Show"}),
    ]

    result, stdout, _stderr = run_with(lambda: iter(icy_markers), MonitorOptions(marker_filter="ad"))

    assert result.reason == "eof"
    assert result.markers_emitted == 2
    assert [item["Classification"] for item in ndjson_lines(stdout)] == [AD_START, AD_END]


def test_json_out_and_db_raw_json_match_emitted_stdout(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"
    json_out = tmp_path / "markers.ndjson"
    markers = [
        marker(tag="#EXT-X-CUE-OUT", timestamp=1.0),
        marker(marker_type="OTHER", tag=None, timestamp=2.0),
        marker(tag="#EXT-X-CUE-IN", timestamp=3.0),
    ]

    result, stdout, _stderr = run_with(
        markers,
        MonitorOptions(
            source_url="fixture://stream?token=secret",
            marker_filter="ad",
            json_out=json_out,
            db_path=db_path,
        ),
    )

    assert result == MonitorResult(reason="eof", markers_seen=3, markers_emitted=2, markers_filtered=1)
    stdout_lines = stdout.getvalue().splitlines()
    assert json_out.read_text(encoding="utf-8").splitlines() == stdout_lines

    conn = sqlite3.connect(db_path)
    rows = conn.execute("select source_url, raw_json from ad_events order by id").fetchall()
    assert rows == [("fixture://stream?token=secret", line) for line in stdout_lines]


def test_marker_type_filter_matches_marker_type_not_classification() -> None:
    scte_marker = marker("SCTE35", tag=None, classification=UNKNOWN)
    id3_marker = marker("ID3", tag=None, classification=UNKNOWN)

    result, stdout, _stderr = run_with([scte_marker, id3_marker], MonitorOptions(marker_filter="id3"))

    assert result == MonitorResult(reason="eof", markers_seen=2, markers_emitted=1, markers_filtered=1)
    assert ndjson_lines(stdout) == [id3_marker.to_dict()]


def test_invalid_filter_fails_fast_with_redacted_error() -> None:
    result, stdout, stderr = run_with([marker()], MonitorOptions(source_url="http://example.test/live?token=secret", marker_filter="bogus"))

    assert result.reason == "error"
    assert result.error is not None
    assert stdout.getvalue() == ""
    assert "[tidemark] error:" in stderr.getvalue()
    assert "invalid marker filter" in stderr.getvalue()
    assert "bogus" not in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()


def test_empty_iterator_returns_eof_count_zero() -> None:
    result, stdout, stderr = run_with([])

    assert result == MonitorResult(reason="eof", markers_seen=0, markers_emitted=0, markers_filtered=0)
    assert stdout.getvalue() == ""
    assert "reason=eof markers=0 emitted=0 filtered=0" in stderr.getvalue()


def test_iterator_error_is_fatal_but_preserves_prior_stdout() -> None:
    first = marker(tag="#EXT-X-CUE-OUT")

    def broken_iter():
        yield first
        raise RuntimeError("boom token=secret raw_base64=abc")

    result, stdout, stderr = run_with(broken_iter(), MonitorOptions(source_url="https://stream.example/path?token=secret"))

    assert result.reason == "error"
    assert result.markers_seen == 1
    assert ndjson_lines(stdout) == [first.to_dict()]
    assert "[tidemark] error:" in stderr.getvalue()
    assert "marker iterator failed" in stderr.getvalue()
    assert "boom" not in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()
    assert "raw_base64" not in stderr.getvalue()


def test_non_marker_item_is_fatal_and_redacted() -> None:
    result, stdout, stderr = run_with([{"url": "http://example.test?token=secret", "raw_base64": "abc"}])

    assert result.reason == "error"
    assert result.markers_seen == 0
    assert stdout.getvalue() == ""
    assert "marker source yielded non-AdMarker value" in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()
    assert "raw_base64" not in stderr.getvalue()


def test_timeout_before_first_marker_returns_timeout_without_consuming() -> None:
    consumed = False

    def source():
        nonlocal consumed
        consumed = True
        yield marker()

    result, stdout, stderr = run_with(source(), MonitorOptions(timeout=0, clock=lambda: 10.0))

    assert result.reason == "timeout"
    assert result.markers_seen == 0
    assert stdout.getvalue() == ""
    assert consumed is False
    assert "reason=timeout markers=0 emitted=0 filtered=0" in stderr.getvalue()


def test_timeout_between_markers_stops_cleanly() -> None:
    times = iter([0.0, 0.0, 2.0])
    first = marker(tag="#EXT-X-CUE-OUT")
    second = marker(tag="#EXT-X-CUE-IN")

    result, stdout, stderr = run_with([first, second], MonitorOptions(timeout=1.0, clock=lambda: next(times)))

    assert result.reason == "timeout"
    assert result.markers_seen == 1
    assert ndjson_lines(stdout) == [first.to_dict()]
    assert "reason=timeout markers=1 emitted=1 filtered=0" in stderr.getvalue()


def test_keyboard_interrupt_stops_cleanly_without_error() -> None:
    first = marker(tag="#EXT-X-CUE-OUT")

    def interrupted_iter():
        yield first
        raise KeyboardInterrupt

    result, stdout, stderr = run_with(interrupted_iter())

    assert result.reason == "interrupted"
    assert result.markers_seen == 1
    assert ndjson_lines(stdout) == [first.to_dict()]
    assert "[tidemark] error:" not in stderr.getvalue()
    assert "reason=interrupted markers=1 emitted=1 filtered=0" in stderr.getvalue()


def test_json_out_open_failure_fails_fast_and_redacts_path(tmp_path: Path) -> None:
    missing_parent_path = tmp_path / "missing" / "markers.ndjson"

    result, stdout, stderr = run_with([marker()], MonitorOptions(json_out=missing_parent_path))

    assert result.reason == "error"
    assert stdout.getvalue() == ""
    assert "json-out setup failed" in stderr.getvalue()
    assert str(missing_parent_path) not in stderr.getvalue()


def test_json_out_per_marker_write_failure_warns_and_continues(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class BrokenWriter(io.StringIO):
        def write(self, value: str) -> int:  # noqa: ARG002
            raise OSError("disk full token=secret")

    handle = BrokenWriter()
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: handle)
    first = marker(tag="#EXT-X-CUE-OUT")
    second = marker(tag="#EXT-X-CUE-IN")

    result, stdout, stderr = run_with([first, second], MonitorOptions(json_out=tmp_path / "markers.ndjson"))

    assert result.reason == "eof"
    assert result.sink_warnings == 2
    assert stdout.getvalue().splitlines() == [first.to_json(), second.to_json()]
    assert stderr.getvalue().count("[tidemark] warning: json-out write failed") == 2
    assert "disk full" not in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()
    assert handle.closed


def test_db_initialize_failure_fails_fast_and_redacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tidemark.monitor as monitor_module

    def fail_initialize(path: object) -> sqlite3.Connection:  # noqa: ARG001
        raise sqlite3.OperationalError("unable to open /secret/path?token=secret")

    monkeypatch.setattr(monitor_module.db, "initialize_db", fail_initialize)

    result, stdout, stderr = run_with([marker()], MonitorOptions(db_path=tmp_path / "events.sqlite"))

    assert result.reason == "error"
    assert stdout.getvalue() == ""
    assert "database setup failed" in stderr.getvalue()
    assert "secret" not in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()


def test_db_per_marker_insert_failure_warns_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    import tidemark.monitor as monitor_module

    conn = sqlite3.connect(":memory:")
    migrate(conn)
    closed = False

    class ClosingConnection:
        def close(self) -> None:
            nonlocal closed
            closed = True
            conn.close()

    def fail_insert(conn: object, source_url: str, marker: AdMarker) -> int:  # noqa: ARG001
        raise sqlite3.OperationalError("insert failed for token=secret")

    monkeypatch.setattr(monitor_module.db, "initialize_db", lambda path: ClosingConnection())
    monkeypatch.setattr(monitor_module.db, "insert_ad_event", fail_insert)
    first = marker(tag="#EXT-X-CUE-OUT")
    second = marker(tag="#EXT-X-CUE-IN")

    result, stdout, stderr = run_with([first, second], MonitorOptions(db_path=":memory:"))

    assert result.reason == "eof"
    assert result.sink_warnings == 2
    assert stdout.getvalue().splitlines() == [first.to_json(), second.to_json()]
    assert stderr.getvalue().count("[tidemark] warning: database write failed") == 2
    assert "insert failed" not in stderr.getvalue()
    assert "token=secret" not in stderr.getvalue()
    assert closed


def test_all_markers_filtered_are_not_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"

    result, stdout, _stderr = run_with(
        [marker(marker_type="OTHER", tag=None)],
        MonitorOptions(marker_filter=AD_START, db_path=db_path),
    )

    assert result == MonitorResult(reason="eof", markers_seen=1, markers_emitted=0, markers_filtered=1)
    assert stdout.getvalue() == ""
    conn = sqlite3.connect(db_path)
    assert conn.execute("select count(*) from ad_events").fetchone()[0] == 0


def test_unexpected_marker_fields_do_not_break_monitor() -> None:
    malformed = marker("ID3", tag=None)
    malformed.fields = {"Frames": [{"Description": object(), "Text": ["promo"]}]}

    result, stdout, stderr = run_with([malformed])

    assert result.reason == "error"
    assert result.markers_seen == 1
    assert stdout.getvalue() == ""
    assert "marker serialization failed" in stderr.getvalue()
    assert "object" not in stderr.getvalue()
