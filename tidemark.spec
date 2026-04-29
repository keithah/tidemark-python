# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import threefive
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

certifi_datas, certifi_binaries, certifi_hiddenimports = collect_all("certifi")
av_datas, av_binaries, av_hiddenimports = collect_all("av")
tidemark_hiddenimports = collect_submodules("tidemark")
tidemark_metadata = copy_metadata("tidemark")
threefive_hiddenimports = collect_submodules("threefive")
threefive_source_root = str(Path(threefive.__file__).resolve().parent.parent)


def optional_collect_all(package_name):
    try:
        return collect_all(package_name)
    except Exception:
        return [], [], []


cocoa_datas, cocoa_binaries, cocoa_hiddenimports = optional_collect_all("Foundation")
speech_datas, speech_binaries, speech_hiddenimports = optional_collect_all("Speech")


a = Analysis(
    ["scripts/tidemark_pyinstaller_entry.py"],
    pathex=["src", threefive_source_root],
    binaries=certifi_binaries + av_binaries + cocoa_binaries + speech_binaries,
    datas=certifi_datas + av_datas + tidemark_metadata + cocoa_datas + speech_datas,
    hiddenimports=(
        tidemark_hiddenimports
        + threefive_hiddenimports
        + certifi_hiddenimports
        + av_hiddenimports
        + cocoa_hiddenimports
        + speech_hiddenimports
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="tidemark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
