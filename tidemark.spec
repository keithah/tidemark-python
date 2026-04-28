# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import threefive
from PyInstaller.utils.hooks import collect_all, collect_submodules

imageio_datas, imageio_binaries, imageio_hiddenimports = collect_all("imageio_ffmpeg")
tidemark_hiddenimports = collect_submodules("tidemark")
threefive_hiddenimports = collect_submodules("threefive")
threefive_source_root = str(Path(threefive.__file__).resolve().parent.parent)


a = Analysis(
    ["scripts/tidemark_pyinstaller_entry.py"],
    pathex=["src", threefive_source_root],
    binaries=imageio_binaries,
    datas=imageio_datas,
    hiddenimports=tidemark_hiddenimports + threefive_hiddenimports + imageio_hiddenimports,
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
