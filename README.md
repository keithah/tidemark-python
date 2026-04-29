# tidemark

SCTE-35 ad marker detection, audio fingerprinting, and transcript search for live radio/TV streams.

## Install

Download the pre-built binary for your platform from the [latest release](https://github.com/keithah/tidemark-python/releases/latest).

> **Note:** If `curl .../releases/latest/download/...` returns a 9-byte file or a "Not found" error, use the pinned tag URL instead: replace `latest/download` with `download/v0.1.0` (or whatever the current version is). GitHub's `latest` redirect can lag a few minutes after a new release.

### Linux (x86_64)

```sh
curl -Lo tidemark https://github.com/keithah/tidemark-python/releases/latest/download/tidemark-linux-x86_64
chmod +x tidemark
sudo mv tidemark /usr/local/bin/
tidemark --help
```

### macOS (Apple Silicon — M1/M2/M3/M4)

```sh
curl -Lo tidemark https://github.com/keithah/tidemark-python/releases/latest/download/tidemark-macos-arm64
chmod +x tidemark
xattr -d com.apple.quarantine tidemark
sudo mv tidemark /usr/local/bin/
tidemark --help
```

macOS will block unsigned binaries on first run. The `xattr` step above clears the quarantine flag before installing. Alternatively, go to **System Settings → Privacy & Security** and click **Allow Anyway** after the first blocked attempt.

> macOS Intel is not supported. Intel Mac users can [build from source](#building-from-source).

### Windows (x86_64)

Download `tidemark-windows-x86_64.exe` from the [latest release](https://github.com/keithah/tidemark-python/releases/latest), rename it to `tidemark.exe`, and place it somewhere on your `PATH` (e.g. add a `C:\tools\` directory to your user `PATH`).

Windows Defender may flag the binary on first run. Click **More info → Run anyway** or add an exclusion in Windows Security settings.

## Quick start

`tidemark ingest` decodes audio with `ffmpeg` from your `PATH`. Install a normal system ffmpeg first, for example `brew install ffmpeg` on macOS.

```sh
# Monitor a live HLS stream and print SCTE-35 markers as JSON
tidemark monitor https://example.com/stream.m3u8

# Same stream, persist to SQLite
tidemark monitor https://example.com/stream.m3u8 --db stream.db

# Ingest a live HLS playlist; defaults to Apple Speech transcription on macOS
tidemark ingest https://example.com/stream.m3u8 --db stream.db

# Search transcript words
tidemark search "traffic report" --db stream.db

# Show status of running monitor/ingest processes
tidemark status

# Generate a song report
tidemark report --db stream.db
```

## Building from source

Requires Python 3.11+.

```sh
git clone https://github.com/keithah/tidemark-python
cd tidemark-python
pip install -e '.[dev]'

# Run tests
pytest

# Build a standalone executable (requires PyInstaller)
pyinstaller tidemark.spec
# Output: dist/tidemark
```

## License

Proprietary.
