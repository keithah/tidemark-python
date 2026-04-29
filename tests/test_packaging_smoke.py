from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import wave
from pathlib import Path

import shutil
import pytest

from tidemark.store import insert_retained_audio, insert_segment, migrate


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "scte35_splice_null.ts"
EXPECTED_HELP_COMMANDS = ("monitor", "ingest", "status", "search", "report", "clip", "doctor")
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
TIMELINE_TABLES = ("segments", "transcript_words", "ad_events", "retained_audio", "songs")


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


def make_tiny_wav(path: Path) -> Path:
    sample_rate = 8000
    n = int(sample_rate * 0.20)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *[int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]))
    path.write_bytes(buf.getvalue())
    return path


def write_manifest(path: Path, segment_name: str) -> Path:
    path.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-MEDIA-SEQUENCE:37",
                "#EXT-X-CUE-OUT:DURATION=15.0",
                "#EXTINF:0.20,",
                segment_name,
                "#EXT-X-CUE-IN",
                "#EXT-X-ENDLIST",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_transcript(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {"text": "hello", "start_offset": 0.03, "end_offset": 0.06, "confidence": 0.9},
                {"text": "tidemark", "start_offset": 0.07, "end_offset": 0.11, "confidence": 0.8},
                {"text": "search", "start_offset": 0.12, "end_offset": 0.16},
            ]
        ),
        encoding="utf-8",
    )
    return path


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def table_counts(db_path: Path, table_names: tuple[str, ...]) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        return {table_name: count_rows(conn, table_name) for table_name in table_names}


def assert_no_traceback_or_private_path(result: subprocess.CompletedProcess[str], *private_paths: Path) -> None:
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, _format_result(result)
    for private_path in private_paths:
        private_text = str(private_path)
        assert private_text not in combined, _format_result(result)


def assert_public_output_redacted(*outputs: str, forbidden: tuple[str, ...]) -> None:
    combined = "\n".join(outputs)
    for value in forbidden:
        assert value not in combined, combined
    assert "Traceback" not in combined, combined
    assert "ACOUSTID" not in combined, combined
    assert "api_key" not in combined.lower(), combined
    assert "secret" not in combined.lower(), combined
    assert "RawBase64" not in combined, combined


def _pcm_s16le(frames: int, *, start: int = 0, channels: int = 1) -> bytes:
    samples: list[bytes] = []
    for frame in range(frames):
        for channel in range(channels):
            value = ((start + frame + channel) % 128) - 64
            samples.append(value.to_bytes(2, "little", signed=True))
    return b"".join(samples)


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _wav_metadata(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def create_retained_audio_clip_fixture(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "private-state" / "tidemark.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    retained_dir = db_path.parent / "audio"
    retained_dir.mkdir()
    retained_path = retained_dir / "secret-fixture-name.wav"
    pcm = _pcm_s16le(8000)
    _write_wav(retained_path, pcm)
    byte_length, sha256 = _wav_metadata(retained_path)

    conn = sqlite3.connect(db_path)
    try:
        migrate(conn)
        segment_id = insert_segment(
            conn,
            source_url="https://example.test/private/source.m3u8?token=secret",
            sequence=37,
            resolved_uri="file:///Users/alice/private/segment.ts?token=secret",
            local_path="/Users/alice/private/segment.ts",
            start_ts=10.0,
            duration_seconds=0.5,
            byte_length=byte_length,
            sha256="0" * 64,
        )
        insert_retained_audio(
            conn,
            segment_id=segment_id,
            source_url="https://example.test/private/source.m3u8?token=secret",
            segment_sequence=37,
            path=retained_path.relative_to(db_path.parent).as_posix(),
            format="wav",
            sample_rate=16000,
            channels=1,
            sample_format="s16le",
            start_ts=10.0,
            duration_seconds=0.5,
            byte_length=byte_length,
            sha256=sha256,
        )
        conn.commit()
    finally:
        conn.close()
    return db_path, retained_path


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


def test_packaged_restart_ingest_skips_existing_segments_and_preserves_timeline_tables(
    packaged_cli: Path,
    clean_env: dict[str, str],
    tmp_path: Path,
) -> None:
    media_path = make_tiny_wav(tmp_path / "private-segment37.wav")
    manifest = write_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    transcript = write_transcript(tmp_path / "private-transcript.json")
    db_path = tmp_path / "private-tidemark.db"
    runtime_dir = tmp_path / "runtime"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text('[paths]\nruntime_dir = "runtime"\n', encoding="utf-8")
    forbidden_public_values = (
        str(tmp_path),
        media_path.name,
        manifest.name,
        transcript.name,
        db_path.name,
        "private",
    )

    first = run_packaged(
        packaged_cli,
        clean_env,
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fixture-transcript",
        transcript.name,
        "--config",
        config_path.name,
        cwd=tmp_path,
        timeout=20.0,
    )

    assert first.returncode == 0, _format_result(first)
    assert first.stderr == "", _format_result(first)
    assert first.stdout == "Ingest complete: segments=1 processed=1 skipped=0 failed=0 words=3 markers=1 issues=0\n"
    assert_public_output_redacted(first.stdout, first.stderr, forbidden=forbidden_public_values)
    counts_after_first = table_counts(db_path, TIMELINE_TABLES)
    assert counts_after_first == {
        "segments": 1,
        "transcript_words": 3,
        "ad_events": 1,
        "retained_audio": 0,
        "songs": 0,
    }

    second = run_packaged(
        packaged_cli,
        clean_env,
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fixture-transcript",
        transcript.name,
        "--config",
        config_path.name,
        cwd=tmp_path,
        timeout=20.0,
    )

    assert second.returncode == 0, _format_result(second)
    assert second.stderr == "", _format_result(second)
    assert second.stdout == "Ingest complete: segments=1 processed=0 skipped=1 failed=0 words=0 markers=0 issues=0\n"
    assert_public_output_redacted(second.stdout, second.stderr, forbidden=forbidden_public_values)
    counts_after_second = table_counts(db_path, TIMELINE_TABLES)
    assert counts_after_second == counts_after_first, {
        table_name: (counts_after_first[table_name], counts_after_second[table_name])
        for table_name in TIMELINE_TABLES
        if counts_after_first[table_name] != counts_after_second[table_name]
    }

    status = run_packaged(
        packaged_cli,
        clean_env,
        "status",
        "--runtime-dir",
        "runtime",
        cwd=tmp_path,
    )

    assert status.returncode == 0, _format_result(status)
    assert status.stderr == "", _format_result(status)
    assert "command=ingest" in status.stdout, _format_result(status)
    assert (
        "counters=failed=0,issues=0,markers=0,processed=0,retained=0,segments=1,skipped=1,songs=0,words=0"
        in status.stdout
    ), _format_result(status)
    assert_public_output_redacted(status.stdout, status.stderr, forbidden=forbidden_public_values)

    search = run_packaged(
        packaged_cli,
        clean_env,
        "search",
        "tidemark search",
        "--db",
        db_path.name,
        "--context",
        "1",
        "--json",
        cwd=tmp_path,
    )

    assert search.returncode == 0, _format_result(search)
    assert search.stderr == "", _format_result(search)
    search_rows = json.loads(search.stdout)
    assert isinstance(search_rows, list), _format_result(search)
    assert len(search_rows) == 1, _format_result(search)
    [search_row] = search_rows
    assert search_row["matched_text"] == "tidemark search"
    assert search_row["context_text"] == "hello tidemark search"
    assert search_row["segment_sequence"] == 37
    assert_public_output_redacted(search.stdout, search.stderr, forbidden=forbidden_public_values)

    ads = run_packaged(
        packaged_cli,
        clean_env,
        "report",
        "ads",
        "--db",
        db_path.name,
        "--json",
        cwd=tmp_path,
    )

    assert ads.returncode == 0, _format_result(ads)
    assert ads.stderr == "", _format_result(ads)
    ad_rows = json.loads(ads.stdout)
    assert isinstance(ad_rows, list), _format_result(ads)
    assert ad_rows, _format_result(ads)
    assert any(row["classification"] == "AD_START" and row["count"] >= 1 for row in ad_rows), _format_result(ads)
    assert_public_output_redacted(ads.stdout, ads.stderr, forbidden=forbidden_public_values)


def test_packaged_clip_exports_from_retained_audio_database(
    packaged_cli: Path,
    clean_env: dict[str, str],
    tmp_path: Path,
) -> None:
    db_path, retained_path = create_retained_audio_clip_fixture(tmp_path)
    db_arg = db_path.relative_to(tmp_path)
    clip_path = tmp_path / "private-exported-clip.wav"
    forbidden_public_values = (
        str(tmp_path),
        db_path.name,
        retained_path.name,
        clip_path.name,
        "private",
    )

    clip = run_packaged(
        packaged_cli,
        clean_env,
        "clip",
        "--at",
        "10.25",
        "--context",
        "0.05",
        "--db",
        db_arg,
        "--out",
        clip_path.name,
        cwd=tmp_path,
    )

    assert clip.returncode == 0, _format_result(clip)
    assert clip.stderr == "", _format_result(clip)
    assert re.fullmatch(r"Clip exported: start=10\.200 duration=0\.100 bytes=\d+ sha256=[0-9a-f]{64}\n", clip.stdout)
    assert_public_output_redacted(clip.stdout, clip.stderr, forbidden=forbidden_public_values)
    assert clip_path.read_bytes().startswith(b"RIFF")

    with wave.open(str(clip_path), "rb") as exported:
        assert exported.getnchannels() == 1
        assert exported.getframerate() == 16000
        assert exported.getsampwidth() == 2
        assert 0 < exported.getnframes() <= 1600


def test_packaged_doctor_emits_check_results_without_traceback(
    packaged_cli: Path,
    clean_env: dict[str, str],
    tmp_path: Path,
) -> None:
    result = run_packaged(packaged_cli, clean_env, "doctor", cwd=tmp_path)

    # No traceback regardless of exit code (exit 1 is valid when checks fail, e.g. no Apple Speech on Linux)
    assert "Traceback" not in result.stdout, _format_result(result)
    assert "Traceback" not in result.stderr, _format_result(result)

    # All expected check labels must appear
    for label in ("python", "version", "audio decoder", "apple speech", "store"):
        assert label in result.stdout, f"missing check label {label!r}\n{_format_result(result)}"

    # Version line must report the correct version
    assert "tidemark 0.1.3" in result.stdout, _format_result(result)

    # PyAV must be available in the packaged binary
    assert "[ok] audio decoder" in result.stdout, _format_result(result)
