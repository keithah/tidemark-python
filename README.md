# tidemark

SCTE-35 ad marker detection, audio fingerprinting, and transcript search for live radio/TV streams.

## Install

Download the pre-built binary for your platform from the [latest release](../../releases/latest).

### Linux (x86_64)

```sh
curl -Lo tidemark https://github.com/keithah/tidemark-python/releases/latest/download/tidemark-linux-x86_64
chmod +x tidemark
sudo mv tidemark /usr/local/bin/
```

### macOS (Apple Silicon — M1/M2/M3)

```sh
curl -Lo tidemark https://github.com/keithah/tidemark-python/releases/latest/download/tidemark-macos-arm64
chmod +x tidemark
sudo mv tidemark /usr/local/bin/
```

macOS will block the binary on first run because it is unsigned. To allow it:

```sh
xattr -d com.apple.quarantine /usr/local/bin/tidemark
```

Or go to **System Settings → Privacy & Security** and click **Allow Anyway** after the first blocked run.

### macOS (Intel)

```sh
curl -Lo tidemark https://github.com/keithah/tidemark-python/releases/latest/download/tidemark-macos-x86_64
chmod +x tidemark
sudo mv tidemark /usr/local/bin/
# clear quarantine if blocked:
xattr -d com.apple.quarantine /usr/local/bin/tidemark
```

### Windows (x86_64)

Download `tidemark-windows-x86_64.exe` from the [latest release](../../releases/latest), rename it to `tidemark.exe`, and move it somewhere on your `PATH` (e.g. `C:\Windows\System32\` or a directory listed in your user `PATH`).

Windows Defender may flag the binary on first run. Click **More info → Run anyway** or add an exclusion in Windows Security.

## Quick start

```sh
# Monitor a live HLS stream and print SCTE-35 markers as JSON
tidemark monitor https://example.com/stream.m3u8

# Same stream, persist to SQLite
tidemark monitor https://example.com/stream.m3u8 --db stream.db

# Ingest a local HLS playlist, transcribe segments, fingerprint songs
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
