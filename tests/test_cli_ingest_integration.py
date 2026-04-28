from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import pytest

from tidemark.audio import decode_segment_audio
from tidemark.fingerprint import FingerprintError, fingerprint_audio_chunk
from tidemark.ingest import resolve_segments
from tidemark.store import SCHEMA_VERSION, initialize_db, insert_fingerprint_cache


CLI = Path.cwd() / ".venv/bin/tidemark"


def run_tidemark(
    *args: object,
    cwd: Path,
    timeout: float = 10.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert CLI.exists(), "expected editable install to provide .venv/bin/tidemark"
    command = [str(CLI), *[str(arg) for arg in args]]
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, cwd=cwd, env=env)


def make_tiny_wav(path: Path) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.20:sample_rate=8000",
            "-ac",
            "1",
            "-y",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
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


def write_repeated_manifest(path: Path, segment_name: str) -> Path:
    path.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-MEDIA-SEQUENCE:37",
                "#EXT-X-CUE-OUT:DURATION=0.40",
                "#EXTINF:0.20,",
                segment_name,
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


def cli_env_without_acoustid_key() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ACOUSTID_API_KEY", None)
    return env


def seed_fingerprint_cache_for_manifest(manifest: Path, db_path: Path) -> None:
    [segment] = resolve_segments(manifest)
    seed_fingerprint_cache_for_segment(segment, db_path)


def seed_fingerprint_cache_for_segment_index(manifest: Path, db_path: Path, index: int = 0) -> None:
    segments = resolve_segments(manifest)
    seed_fingerprint_cache_for_segment(segments[index], db_path)


def seed_fingerprint_cache_for_segment(segment, db_path: Path) -> None:
    chunk = decode_segment_audio(segment)
    try:
        fingerprint = fingerprint_audio_chunk(chunk)
    except FingerprintError as exc:
        pytest.skip(f"real acoustid/Chromaprint backend unavailable: {exc}")

    with initialize_db(db_path) as conn:
        insert_fingerprint_cache(
            conn,
            fingerprint=fingerprint.fingerprint,
            acoustid_id="acoustid-public-cache-hit",
            recording_id="recording-public-cache-hit",
            title="Generated integration tone",
            artist="Tidemark integration tests",
            album="Offline cache evidence",
            score=0.97,
            raw_status="ok",
            lookup_source="seeded-installed-cli-cache",
        )


def assert_public_output_redacted(*outputs: str, forbidden: tuple[str, ...]) -> None:
    combined = "\n".join(outputs)
    for value in forbidden:
        assert value not in combined
    assert "ACOUSTID" not in combined
    assert "api_key" not in combined.lower()
    assert "secret" not in combined.lower()
    assert "Traceback" not in combined


def table_counts(db_path: Path, table_names: tuple[str, ...]) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        return {table_name: count_rows(conn, table_name) for table_name in table_names}


def assert_installed_fingerprint_ingest_succeeds(
    *,
    tmp_path: Path,
    manifest: Path,
    media_path: Path,
    db_path: Path,
) -> subprocess.CompletedProcess[str]:
    ingest = run_tidemark(
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fingerprint",
        cwd=tmp_path,
        env=cli_env_without_acoustid_key(),
    )

    assert ingest.returncode == 0, ingest.stderr
    assert ingest.stderr == ""
    assert "segments=1" in ingest.stdout
    assert "words=0" in ingest.stdout
    assert "markers=1" in ingest.stdout
    assert "retained=1" in ingest.stdout
    assert "songs=1" in ingest.stdout
    assert "issues=0" in ingest.stdout
    assert str(tmp_path) not in ingest.stdout
    assert media_path.name not in ingest.stdout
    assert manifest.name not in ingest.stdout
    assert db_path.name not in ingest.stdout
    assert "ACOUSTID" not in ingest.stdout
    assert "api_key" not in ingest.stdout.lower()
    assert "secret" not in ingest.stdout.lower()
    return ingest


def test_installed_restart_ingest_skips_existing_segments_and_preserves_timeline_tables(tmp_path: Path) -> None:
    media_path = make_tiny_wav(tmp_path / "segment37.wav")
    manifest = write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    transcript = write_transcript(tmp_path / "transcript.json")
    db_path = tmp_path / "tidemark.db"
    runtime_dir = tmp_path / "runtime"
    config_path = tmp_path / "tidemark.toml"
    config_path.write_text(f'[paths]\nruntime_dir = "{runtime_dir}"\n', encoding="utf-8")
    timeline_tables = ("segments", "transcript_words", "ad_events", "retained_audio", "songs")

    seed_fingerprint_cache_for_manifest(manifest, db_path)

    first = run_tidemark(
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fixture-transcript",
        transcript.name,
        "--fingerprint",
        "--config",
        config_path.name,
        cwd=tmp_path,
        env=cli_env_without_acoustid_key(),
    )

    assert first.returncode == 0, f"stdout={first.stdout}\nstderr={first.stderr}"
    assert first.stderr == ""
    assert first.stdout == (
        "Ingest complete: segments=1 processed=1 skipped=0 failed=0 "
        "words=3 markers=1 retained=1 songs=1 issues=0\n"
    )
    counts_after_first = table_counts(db_path, timeline_tables)
    assert counts_after_first == {
        "segments": 1,
        "transcript_words": 3,
        "ad_events": 1,
        "retained_audio": 1,
        "songs": 1,
    }

    second = run_tidemark(
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fixture-transcript",
        transcript.name,
        "--fingerprint",
        "--config",
        config_path.name,
        cwd=tmp_path,
        env=cli_env_without_acoustid_key(),
    )

    assert second.returncode == 0, f"stdout={second.stdout}\nstderr={second.stderr}"
    assert second.stderr == ""
    assert second.stdout == (
        "Ingest complete: segments=1 processed=0 skipped=1 failed=0 "
        "words=0 markers=0 retained=0 songs=0 issues=0\n"
    )
    counts_after_second = table_counts(db_path, timeline_tables)
    assert counts_after_second == counts_after_first, {
        table_name: (counts_after_first[table_name], counts_after_second[table_name])
        for table_name in timeline_tables
        if counts_after_first[table_name] != counts_after_second[table_name]
    }

    status = run_tidemark("status", "--runtime-dir", runtime_dir, cwd=tmp_path)

    assert status.returncode == 0, f"stdout={status.stdout}\nstderr={status.stderr}"
    assert status.stderr == ""
    assert "command=ingest" in status.stdout
    assert (
        "counters=failed=0,issues=0,markers=0,processed=0,retained=0,segments=1,skipped=1,songs=0,words=0"
        in status.stdout
    )
    assert str(tmp_path) not in status.stdout
    assert "Traceback" not in status.stdout


def test_installed_fingerprint_ingest_uses_seeded_cache_without_transcript_or_api_key(tmp_path: Path) -> None:
    media_path = make_tiny_wav(tmp_path / "private-segment37.wav")
    manifest = write_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.db"

    seed_fingerprint_cache_for_manifest(manifest, db_path)
    assert_installed_fingerprint_ingest_succeeds(tmp_path=tmp_path, manifest=manifest, media_path=media_path, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        ad_marker_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM ad_events
            WHERE classification = 'AD_START' AND source = 'hls_manifest'
            """
        ).fetchone()[0]
        retained_path_text = conn.execute("SELECT path FROM retained_audio").fetchone()[0]
        retained_columns = conn.execute("PRAGMA table_info(retained_audio)").fetchall()

        assert user_version == SCHEMA_VERSION
        assert count_rows(conn, "segments") == 1
        assert count_rows(conn, "transcript_words") == 0
        assert ad_marker_count == 1
        assert count_rows(conn, "fingerprint_cache") == 1
        assert count_rows(conn, "songs") == 1
        assert count_rows(conn, "retained_audio") == 1
        assert all(column[2].upper() != "BLOB" for column in retained_columns)

    retained_path = Path(retained_path_text)
    retained_file = retained_path if retained_path.is_absolute() else tmp_path / retained_path
    assert retained_file.exists()
    assert retained_file.parent == db_path.parent / "tidemark-audio"
    assert retained_file.suffix == ".wav"
    assert media_path.stem not in retained_file.name
    assert "private" not in retained_file.name


def test_installed_m003_flow_ingests_reports_and_exports_clip_from_one_database(tmp_path: Path) -> None:
    media_path = make_tiny_wav(tmp_path / "private-segment37.wav")
    manifest = write_repeated_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.db"
    clip_path = tmp_path / "private-exported-clip.wav"
    forbidden_public_values = (str(tmp_path), media_path.name, manifest.name, db_path.name, clip_path.name, "private")

    seed_fingerprint_cache_for_segment_index(manifest, db_path, index=0)

    ingest = run_tidemark(
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fingerprint",
        cwd=tmp_path,
        env=cli_env_without_acoustid_key(),
    )

    assert ingest.returncode == 0, f"stdout={ingest.stdout}\nstderr={ingest.stderr}"
    assert ingest.stderr == ""
    assert "segments=2" in ingest.stdout
    assert "words=0" in ingest.stdout
    assert "markers=1" in ingest.stdout
    assert "retained=2" in ingest.stdout
    assert "songs=2" in ingest.stdout
    assert "issues=0" in ingest.stdout
    assert_public_output_redacted(ingest.stdout, ingest.stderr, forbidden=forbidden_public_values)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert count_rows(conn, "segments") == 2
        assert count_rows(conn, "transcript_words") == 0
        assert count_rows(conn, "fingerprint_cache") == 1
        assert count_rows(conn, "songs") == 2
        assert count_rows(conn, "retained_audio") == 2
        assert (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM ad_events
                WHERE classification = 'AD_START' AND source = 'hls_manifest'
                """
            ).fetchone()[0]
            == 1
        )
        retained_paths = [row[0] for row in conn.execute("SELECT path FROM retained_audio ORDER BY start_ts").fetchall()]

    for retained_path_text in retained_paths:
        retained_path = Path(retained_path_text)
        retained_file = retained_path if retained_path.is_absolute() else tmp_path / retained_path
        assert retained_file.exists(), retained_file
        assert retained_file.parent == db_path.parent / "tidemark-audio"
        assert retained_file.suffix == ".wav"
        assert media_path.stem not in retained_file.name
        assert "private" not in retained_file.name

    plays = run_tidemark("report", "plays", "--db", db_path.name, "--json", cwd=tmp_path)
    repeats = run_tidemark("report", "repeats", "--db", db_path.name, "--json", cwd=tmp_path)
    ads = run_tidemark("report", "ads", "--db", db_path.name, "--json", cwd=tmp_path)

    for result in (plays, repeats, ads):
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert result.stderr == ""
        assert_public_output_redacted(result.stdout, result.stderr, forbidden=forbidden_public_values)

    play_rows = json.loads(plays.stdout)
    assert isinstance(play_rows, list)
    assert len(play_rows) == 2
    assert [row["title"] for row in play_rows] == ["Generated integration tone", "Generated integration tone"]
    assert {row["lookup_source"] for row in play_rows} == {"cache"}
    assert {row["segment_sequence"] for row in play_rows} == {37, 38}
    assert [row["start_ts"] for row in play_rows] == [0.0, 0.2]

    repeat_rows = json.loads(repeats.stdout)
    assert isinstance(repeat_rows, list)
    assert len(repeat_rows) == 1
    [repeat_row] = repeat_rows
    assert repeat_row["title"] == "Generated integration tone"
    assert repeat_row["count"] == 2
    assert repeat_row["first_start_ts"] == 0.0
    assert repeat_row["last_start_ts"] == 0.2

    ad_rows = json.loads(ads.stdout)
    assert isinstance(ad_rows, list)
    assert ad_rows
    assert any(row["classification"] == "AD_START" and row["count"] >= 1 for row in ad_rows)

    clip = run_tidemark(
        "clip",
        "--at",
        "0.25",
        "--context",
        "0.05",
        "--db",
        db_path.name,
        "--out",
        clip_path.name,
        cwd=tmp_path,
    )

    assert clip.returncode == 0, f"stdout={clip.stdout}\nstderr={clip.stderr}"
    assert clip.stderr == ""
    assert re.fullmatch(r"Clip exported: start=0\.200 duration=0\.100 bytes=\d+ sha256=[0-9a-f]{64}\n", clip.stdout)
    assert_public_output_redacted(clip.stdout, clip.stderr, forbidden=forbidden_public_values)
    assert clip_path.read_bytes().startswith(b"RIFF")

    with wave.open(str(clip_path), "rb") as exported:
        assert exported.getnchannels() == 1
        assert exported.getframerate() == 16000
        assert exported.getsampwidth() == 2
        assert 0 < exported.getnframes() <= 1600


def test_installed_fingerprint_ingest_retained_audio_can_be_exported_by_clip(tmp_path: Path) -> None:
    media_path = make_tiny_wav(tmp_path / "private-segment37.wav")
    manifest = write_manifest(tmp_path / "private-playlist.m3u8", media_path.name)
    db_path = tmp_path / "tidemark.db"
    clip_path = tmp_path / "private-exported-clip.wav"
    missing_clip_path = tmp_path / "private-missing-clip.wav"

    seed_fingerprint_cache_for_manifest(manifest, db_path)
    assert_installed_fingerprint_ingest_succeeds(tmp_path=tmp_path, manifest=manifest, media_path=media_path, db_path=db_path)

    clip = run_tidemark(
        "clip",
        "--at",
        "0.10",
        "--context",
        "0.05",
        "--db",
        db_path.name,
        "--out",
        clip_path.name,
        cwd=tmp_path,
    )

    assert clip.returncode == 0, clip.stderr
    assert clip.stderr == ""
    assert re.fullmatch(r"Clip exported: start=0\.050 duration=0\.100 bytes=\d+ sha256=[0-9a-f]{64}\n", clip.stdout)
    assert str(tmp_path) not in clip.stdout
    assert media_path.name not in clip.stdout
    assert manifest.name not in clip.stdout
    assert db_path.name not in clip.stdout
    assert clip_path.name not in clip.stdout
    assert "private" not in clip.stdout
    assert clip_path.read_bytes().startswith(b"RIFF")

    with wave.open(str(clip_path), "rb") as exported:
        assert exported.getnchannels() == 1
        assert exported.getframerate() == 16000
        assert exported.getsampwidth() == 2
        assert 0 < exported.getnframes() <= 1600

    missing = run_tidemark(
        "clip",
        "--at",
        "999",
        "--context",
        "0.05",
        "--db",
        db_path.name,
        "--out",
        missing_clip_path.name,
        cwd=tmp_path,
    )

    assert missing.returncode == 1
    assert missing.stdout == ""
    assert not missing_clip_path.exists()
    assert "[tidemark] error: no retained audio covers timestamp" in missing.stderr
    assert "Traceback" not in missing.stderr
    assert str(tmp_path) not in missing.stderr
    assert media_path.name not in missing.stderr
    assert manifest.name not in missing.stderr
    assert db_path.name not in missing.stderr
    assert clip_path.name not in missing.stderr
    assert missing_clip_path.name not in missing.stderr
    assert "private" not in missing.stderr
    assert "ACOUSTID" not in missing.stderr
    assert "api_key" not in missing.stderr.lower()


def test_installed_plain_ingest_without_fixture_remains_nonzero_and_redacted(tmp_path: Path) -> None:
    media_path = make_tiny_wav(tmp_path / "private-segment37.wav")
    manifest = write_manifest(tmp_path / "private-playlist.m3u8", media_path.name)

    ingest = run_tidemark("ingest", manifest.name, cwd=tmp_path)

    assert ingest.returncode != 0
    assert ingest.stdout == ""
    assert "--fixture-transcript is required unless --fingerprint is enabled" in ingest.stderr
    assert "Traceback" not in ingest.stderr
    assert str(tmp_path) not in ingest.stderr
    assert media_path.name not in ingest.stderr
    assert manifest.name not in ingest.stderr


def test_installed_ingest_and_search_share_one_tmp_database(tmp_path: Path) -> None:
    media_path = make_tiny_wav(tmp_path / "segment37.wav")
    manifest = write_manifest(tmp_path / "playlist.m3u8", media_path.name)
    transcript = write_transcript(tmp_path / "transcript.json")
    db_path = tmp_path / "tidemark.db"

    ingest = run_tidemark(
        "ingest",
        manifest.name,
        "--db",
        db_path.name,
        "--fixture-transcript",
        transcript.name,
        cwd=tmp_path,
    )

    assert ingest.returncode == 0, ingest.stderr
    assert ingest.stderr == ""
    assert "segments=1" in ingest.stdout
    assert "words=3" in ingest.stdout
    assert "markers=1" in ingest.stdout
    assert "issues=0" in ingest.stdout

    search = run_tidemark(
        "search",
        "tidemark search",
        "--db",
        db_path.name,
        "--context",
        "1",
        "--json",
        cwd=tmp_path,
    )

    assert search.returncode == 0, search.stderr
    assert search.stderr == ""
    rows = json.loads(search.stdout)
    assert isinstance(rows, list)
    assert len(rows) == 1
    [row] = rows
    assert row["matched_text"] == "tidemark search"
    assert row["context_text"] == "hello tidemark search"
    assert row["segment_sequence"] == 37
    assert row["hit_start_ts"] == 0.07
    assert row["hit_end_ts"] == 0.16
    assert row["context_start_ts"] == 0.03
    assert row["context_end_ts"] == 0.16
    assert len(row["word_ids"]) == 2

    missing = run_tidemark("search", "missing phrase", "--db", db_path.name, "--json", cwd=tmp_path)

    assert missing.returncode == 0, missing.stderr
    assert missing.stderr == ""
    assert json.loads(missing.stdout) == []

    with sqlite3.connect(db_path) as conn:
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        segment_count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        word_count = conn.execute("SELECT COUNT(*) FROM transcript_words").fetchone()[0]
        ad_marker_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM ad_events
            WHERE classification = 'AD_START' AND source = 'hls_manifest'
            """
        ).fetchone()[0]

    assert user_version == SCHEMA_VERSION
    assert segment_count == 1
    assert word_count == 3
    assert ad_marker_count >= 1
    assert count_rows(conn, "fingerprint_cache") == 0
    assert count_rows(conn, "songs") == 0
    assert count_rows(conn, "retained_audio") == 0
    assert not (db_path.parent / "tidemark-audio").exists()
