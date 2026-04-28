from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_EMPTY_OPTIONAL_VALUES = (None, [], {})
_PYTHON_ONLY_SCTE35_KEYS = {"RawBase64", "Command", "Descriptors"}


@dataclass(frozen=True)
class CommandResult:
    label: str
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def assert_success(self) -> None:
        assert not self.timed_out, _format_failure(
            self.label,
            self.returncode,
            self.stderr,
            prefix="command timed out",
        )
        assert self.returncode == 0, _format_failure(self.label, self.returncode, self.stderr)


def python_cli_path() -> Path:
    path = Path(".venv/bin/tidemark")
    assert path.exists(), "expected editable install to provide .venv/bin/tidemark"
    return path


def build_go_cli(tmp_path: Path) -> Path:
    go_source = Path("../tidemark-go")
    assert go_source.exists(), "expected local ../tidemark-go reference checkout"
    output = tmp_path / "tidemark-go"
    result = run_command("build go reference cli", ["go", "build", "-o", str(output), "."], cwd=go_source)
    result.assert_success()
    assert output.exists(), "go build completed without producing tidemark-go binary"
    return output


def run_command(
    label: str,
    args: list[str] | tuple[str, ...],
    *,
    timeout: float = 30.0,
    cwd: Path | None = None,
) -> CommandResult:
    command = [str(arg) for arg in args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            label=_sanitize_text(label),
            args=[_sanitize_text(arg) for arg in command],
            returncode=-1,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
        )

    return CommandResult(
        label=_sanitize_text(label),
        args=[_sanitize_text(arg) for arg in command],
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_ndjson(stdout: str, *, label: str = "ndjson") -> list[dict[str, Any]]:
    if not stdout:
        return []

    markers: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{_sanitize_text(label)} malformed NDJSON on line {line_number}: {exc.msg}"
            ) from exc
        assert isinstance(parsed, dict), (
            f"{_sanitize_text(label)} malformed NDJSON on line {line_number}: expected object"
        )
        markers.append(parsed)
    return markers


def normalize_marker(marker: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in marker.items():
        if key == "Timestamp":
            continue
        if key in _PYTHON_ONLY_SCTE35_KEYS:
            continue
        if value in _EMPTY_OPTIONAL_VALUES:
            continue
        normalized[key] = value
    return normalized


def normalize_markers(markers: list[Mapping[str, Any]], *, sort: bool = False) -> list[dict[str, Any]]:
    normalized = [normalize_marker(marker) for marker in markers]
    if not sort:
        return normalized
    return sorted(normalized, key=lambda marker: json.dumps(marker, sort_keys=True, separators=(",", ":")))


@contextmanager
def serve_http_fixture(routes: Mapping[str, bytes | tuple[bytes, str]]) -> Iterator[str]:
    normalized_routes: dict[str, tuple[bytes, str]] = {}
    for path, payload in routes.items():
        assert path.startswith("/"), f"fixture route must start with '/': {path}"
        if isinstance(payload, tuple):
            body, content_type = payload
        else:
            body, content_type = payload, "application/octet-stream"
        normalized_routes[path] = (body, content_type)

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = normalized_routes.get(urlsplit(self.path).path)
            if route is None:
                self.send_error(404)
                return
            body, content_type = route
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _format_failure(label: str, exit_code: int, stderr: str, *, prefix: str = "command failed") -> str:
    stderr_tail = "\n".join(_sanitize_text(stderr).splitlines()[-20:])
    return f"{prefix}: label={_sanitize_text(label)!r} exit={exit_code} stderr_tail={stderr_tail!r}"


def _sanitize_text(text: object) -> str:
    raw = str(text)
    parts = raw.split()
    sanitized_parts = []
    for part in parts:
        split = urlsplit(part)
        if split.scheme in {"http", "https"} and split.netloc:
            sanitized_parts.append(urlunsplit((split.scheme, split.netloc, split.path, "", split.fragment)))
        else:
            sanitized_parts.append(part)
    return " ".join(sanitized_parts)


__all__ = [
    "CommandResult",
    "build_go_cli",
    "normalize_marker",
    "normalize_markers",
    "parse_ndjson",
    "python_cli_path",
    "run_command",
    "serve_http_fixture",
]
