# -*- mode: python ; coding: utf-8 -*-
# Fabrique PyPhantom.exe : pyinstaller PyPhantom.spec
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
for pkg in ("pyqtgraph", "sounddevice", "PySide6"):
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

# DLL de l'Airspy R2 (pilotées en ctypes), à embarquer à côté de l'exe
for dll in ("airspy.dll", "libusb-1.0.dll", "pthreadVC2.dll", "pthreadVCE2.dll", "libwinpthread-1.dll"):
    import os
    if os.path.exists(dll):
        binaries.append((dll, "."))

a = Analysis(
    ["run_pyphantom.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["radiolab.dsp", "pyphantom.decoders"],
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
    [],
    exclude_binaries=True,
    name="PyPhantom",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PyPhantom",
)
