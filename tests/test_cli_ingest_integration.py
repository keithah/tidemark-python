from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import imageio_ffmpeg


CLI = Path.cwd() / ".venv/bin/tidemark"


def run_tidemark(*args: object, cwd: Path, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    assert CLI.exists(), "expected editable install to provide .venv/bin/tidemark"
    command = [str(CLI), *[str(arg) for arg in args]]
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, cwd=cwd)


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

    assert user_version == 3
    assert segment_count == 1
    assert word_count == 3
    assert ad_marker_count >= 1
