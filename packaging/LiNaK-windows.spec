# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"

pipeline_modules = [
    f"linak.pipelines.{name}"
    for name in (
        "runorca", "orca_to_json", "rydberg", "transitions", "lifetimes",
        "polarizability", "blackbody", "tweezer", "hyperfine", "feshbach",
        "plotinteractive", "spectra", "orbital3d", "compare_elements",
        "scattering_rate",
    )
]

data_sources = [
    "NIST Levels",
    "f_values",
    "Moments",
    "vendor",
    "HTMLs",
    "data_json",
    "Basis sets",
]
datas = [
    (str(ROOT / "resources" / name), f"resources/{name}")
    for name in data_sources
]

analysis = Analysis(
    [str(SRC / "linak" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=pipeline_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LiNaK",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="LiNaK",
)
