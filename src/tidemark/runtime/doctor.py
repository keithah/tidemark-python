"""Preflight checks for tidemark runtime environment."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DoctorResult:
    label: str
    ok: bool
    detail: str


def _check_av() -> DoctorResult:
    try:
        import av  # type: ignore[import-untyped]

        return DoctorResult("audio decoder (PyAV)", ok=True, detail=f"av {av.__version__}")
    except ImportError as exc:
        return DoctorResult("audio decoder (PyAV)", ok=False, detail=str(exc))


def _check_apple_speech() -> DoctorResult:
    if platform.system() != "Darwin":
        return DoctorResult("apple speech", ok=False, detail="not available (requires macOS)")

    try:
        import Speech  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return DoctorResult("apple speech", ok=False, detail="Speech framework not importable (install PyObjC)")

    try:
        from Speech import SFSpeechRecognizer  # type: ignore[import-untyped]

        recognizer = SFSpeechRecognizer.alloc().init()
        if recognizer is None:
            return DoctorResult("apple speech", ok=False, detail="SFSpeechRecognizer unavailable on this device")
        auth_status = SFSpeechRecognizer.authorizationStatus()
        # 0=notDetermined 1=denied 2=restricted 3=authorized
        status_names = {0: "notDetermined", 1: "denied", 2: "restricted", 3: "authorized"}
        status_str = status_names.get(int(auth_status), str(auth_status))
        ok = int(auth_status) == 3
        return DoctorResult("apple speech", ok=ok, detail=f"SFSpeechRecognizer available, authorization: {status_str}")
    except Exception as exc:
        return DoctorResult("apple speech", ok=False, detail=f"error probing SFSpeechRecognizer: {exc}")


def _check_sqlite() -> DoctorResult:
    try:
        import sqlite3

        ver = sqlite3.sqlite_version
        return DoctorResult("store (SQLite)", ok=True, detail=f"SQLite {ver}")
    except Exception as exc:
        return DoctorResult("store (SQLite)", ok=False, detail=str(exc))


def _check_version() -> DoctorResult:
    try:
        from importlib.metadata import PackageNotFoundError, version

        ver = version("tidemark")
        return DoctorResult("version", ok=True, detail=f"tidemark {ver}")
    except Exception:
        from tidemark import __version__

        return DoctorResult("version", ok=True, detail=f"tidemark {__version__} (metadata unavailable)")


def _check_python() -> DoctorResult:
    ver = sys.version.split()[0]
    return DoctorResult("python", ok=True, detail=f"{ver} ({platform.machine()})")


def run_checks() -> list[DoctorResult]:
    return [
        _check_python(),
        _check_version(),
        _check_av(),
        _check_apple_speech(),
        _check_sqlite(),
    ]


__all__ = ["DoctorResult", "run_checks"]
