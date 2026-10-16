# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path(SPECPATH).parent
pipeline_modules = [
    f"linak.pipelines.{name}"
    for name in (
        "runorca", "orca_to_json", "rydberg", "transitions", "lifetimes",
        "polarizability", "blackbody", "tweezer", "hyperfine", "feshbach",
        "plotinteractive", "spectra", "orbital3d", "compare_elements",
        "scattering_rate",
    )
]
datas = [
    (str(ROOT / "resources" / "NIST Levels"), "resources/NIST Levels"),
    (str(ROOT / "resources" / "f_values"), "resources/f_values"),
    (str(ROOT / "resources" / "Moments"), "resources/Moments"),
    (str(ROOT / "resources" / "vendor"), "resources/vendor"),
    (str(ROOT / "resources" / "HTMLs"), "resources/HTMLs"),
    (str(ROOT / "resources" / "data_json"), "resources/data_json"),
    (str(ROOT / "resources" / "Basis sets"), "resources/Basis sets"),
    (str(ROOT / "resources" / "plots"), "resources/plots"),
    (str(ROOT / "resources" / "orca_outputs"), "resources/orca_outputs"),
]

a = Analysis(
    ["../src/linak/__main__.py"],
    pathex=["../src"],
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
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LiNaK",
)
app = BUNDLE(
    coll,
    name="LiNaK.app",
    icon=None,
    bundle_identifier="org.linak.app",
)
