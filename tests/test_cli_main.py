from __future__ import annotations

from typer.testing import CliRunner

from tidemark.cli.main import app
from tidemark.monitor_sources import _open_http_response


runner = CliRunner()


def test_version_falls_back_to_package_constant_when_metadata_is_absent(monkeypatch) -> None:
    from tidemark import __version__
    from tidemark.cli import main as cli_main

    def missing_version(_: str) -> str:
        raise cli_main.PackageNotFoundError

    monkeypatch.setattr(cli_main, "_pkg_version", missing_version)

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"tidemark {__version__}\n"


def test_open_http_response_uses_certifi_ssl_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    response = object()

    def fake_context() -> object:
        return object()

    def fake_urlopen(request, *, timeout, context):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        captured["context"] = context
        return response

    monkeypatch.setattr("tidemark.monitor_sources._ssl_context", fake_context)
    monkeypatch.setattr("tidemark.monitor_sources.urlopen", fake_urlopen)

    actual = _open_http_response(
        "https://example.test/live.m3u8?token=secret",
        timeout=4.5,
        headers={"User-Agent": "tidemark-test"},
    )

    assert actual is response
    assert captured["url"] == "https://example.test/live.m3u8?token=secret"
    assert captured["headers"] == {"User-agent": "tidemark-test"}
    assert captured["timeout"] == 4.5
    assert captured["context"] is not None
