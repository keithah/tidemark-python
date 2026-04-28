import hashlib
from pathlib import Path

import pytest

from tidemark.ingest import (
    SegmentIngestError,
    resolve_local_hls_segments,
    resolve_local_media_file,
    resolve_segments,
)


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_resolve_local_hls_segments_records_sequence_path_time_hash_and_lazy_bytes(tmp_path):
    first_bytes = b"first deterministic segment"
    second_bytes = b"second deterministic segment"
    write_bytes(tmp_path / "media" / "seg10.ts", first_bytes)
    write_bytes(tmp_path / "media" / "seg11.ts", second_bytes)
    manifest = tmp_path / "playlist.m3u8"
    manifest.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-MEDIA-SEQUENCE:10",
                "#EXTINF:4.25,",
                "media/seg10.ts",
                "#EXTINF:5.5,",
                "media/seg11.ts",
            ]
        ),
        encoding="utf-8",
    )

    records = resolve_local_hls_segments(manifest, source_url="fixture://local/private-feed.m3u8?token=secret")

    assert [record.sequence for record in records] == [10, 11]
    assert [record.start_ts for record in records] == [0.0, 4.25]
    assert [record.duration_seconds for record in records] == [4.25, 5.5]
    assert [record.byte_length for record in records] == [len(first_bytes), len(second_bytes)]
    assert [record.sha256 for record in records] == [sha256_hex(first_bytes), sha256_hex(second_bytes)]
    assert [record.source_url for record in records] == ["fixture://local/private-feed.m3u8?token=secret"] * 2
    assert [record.resolved_uri for record in records] == [
        (tmp_path / "media" / "seg10.ts").as_uri(),
        (tmp_path / "media" / "seg11.ts").as_uri(),
    ]
    assert [record.local_path for record in records] == [
        str(tmp_path / "media" / "seg10.ts"),
        str(tmp_path / "media" / "seg11.ts"),
    ]
    assert records[0].metadata == {"manifest_path": str(manifest), "manifest_uri": manifest.as_uri()}
    assert records[0].load_bytes() == first_bytes
    assert records[1].load_bytes() == second_bytes


def test_resolve_local_hls_segments_accumulates_decimal_durations_and_high_media_sequence(tmp_path):
    write_bytes(tmp_path / "a.ts", b"a")
    write_bytes(tmp_path / "b.ts", b"bb")
    manifest = tmp_path / "playlist.m3u8"
    manifest.write_text(
        "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:900\n#EXTINF:0.333,\na.ts\n#EXTINF:0.667,\nb.ts\n",
        encoding="utf-8",
    )

    records = resolve_segments(manifest)

    assert [(record.sequence, record.start_ts, record.duration_seconds) for record in records] == [
        (900, 0.0, 0.333),
        (901, 0.333, 0.667),
    ]


def test_resolve_local_media_file_records_one_segment_with_default_and_provided_source(tmp_path):
    media_bytes = b"single local media file"
    media_path = write_bytes(tmp_path / "direct.ts", media_bytes)

    default_records = resolve_local_media_file(media_path)
    provided_records = resolve_local_media_file(
        media_path,
        source_url="fixture://direct/source.ts?token=secret",
        duration_seconds=12.5,
    )

    assert len(default_records) == 1
    default_record = default_records[0]
    assert default_record.sequence == 0
    assert default_record.source_url == media_path.as_uri()
    assert default_record.resolved_uri == media_path.as_uri()
    assert default_record.local_path == str(media_path)
    assert default_record.start_ts == 0.0
    assert default_record.duration_seconds is None
    assert default_record.byte_length == len(media_bytes)
    assert default_record.sha256 == sha256_hex(media_bytes)
    assert default_record.load_bytes() == media_bytes

    provided_record = provided_records[0]
    assert provided_record.source_url == "fixture://direct/source.ts?token=secret"
    assert provided_record.duration_seconds == 12.5


@pytest.mark.parametrize(
    "manifest_text, expected_phase",
    [
        ("", "manifest"),
        ("#EXTM3U\n#EXTINF:not-a-duration,\nsegment.ts\n", "manifest"),
        ("#EXTM3U\n#EXTINF:4.0,\n", "manifest"),
    ],
)
def test_resolve_local_hls_segments_rejects_malformed_manifests_with_redacted_context(
    tmp_path, manifest_text, expected_phase
):
    manifest = tmp_path / "private-playlist.m3u8"
    manifest.write_text(manifest_text, encoding="utf-8")

    with pytest.raises(SegmentIngestError) as exc_info:
        resolve_local_hls_segments(manifest, source_url="fixture://private/source.m3u8?token=secret")

    message = str(exc_info.value)
    assert expected_phase in message
    assert "token=secret" not in message
    assert "private-playlist" not in message
    assert "private/source" not in message


def test_resolve_local_hls_segments_rejects_missing_segment_file_without_private_names(tmp_path):
    manifest = tmp_path / "private-playlist.m3u8"
    manifest.write_text("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:7\n#EXTINF:4.0,\nprivate-segment.ts\n", encoding="utf-8")

    with pytest.raises(SegmentIngestError) as exc_info:
        resolve_local_hls_segments(manifest, source_url="fixture://private/source.m3u8?token=secret")

    message = str(exc_info.value)
    assert "segment" in message
    assert "sequence 7" in message
    assert "token=secret" not in message
    assert "private-segment" not in message
    assert "private-playlist" not in message


@pytest.mark.parametrize("input_value", ["https://example.test/live/playlist.m3u8?token=secret", "http://example.test/live.ts"])
def test_resolve_segments_rejects_network_urls_for_s01_without_url_leak(input_value):
    with pytest.raises(SegmentIngestError) as exc_info:
        resolve_segments(input_value)

    message = str(exc_info.value)
    assert "unsupported" in message
    assert "source" in message
    assert "example.test" not in message
    assert "token=secret" not in message


def test_segment_record_load_bytes_rejects_changed_content_without_leaking_bytes(tmp_path):
    media_path = write_bytes(tmp_path / "private-direct.ts", b"original bytes")
    record = resolve_local_media_file(media_path, source_url="fixture://direct/source.ts?token=secret")[0]
    media_path.write_bytes(b"changed private bytes")

    with pytest.raises(SegmentIngestError) as exc_info:
        record.load_bytes()

    message = str(exc_info.value)
    assert "load" in message
    assert "sequence 0" in message
    assert "SHA-256" in message or "byte length" in message
    assert "changed private bytes" not in message
    assert "token=secret" not in message
    assert "private-direct" not in message


def test_segment_record_load_bytes_rejects_non_bytes_loader_without_value_leak(tmp_path):
    media_path = write_bytes(tmp_path / "direct.ts", b"bytes")
    record = resolve_local_media_file(media_path)[0]
    bad_record = record.with_loader(lambda: "private text")

    with pytest.raises(SegmentIngestError) as exc_info:
        bad_record.load_bytes()

    message = str(exc_info.value)
    assert "bytes" in message
    assert "sequence 0" in message
    assert "private text" not in message
