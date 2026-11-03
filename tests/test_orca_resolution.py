"""ORCA lookup: real ORCA installs pass, a screen reader called orca does not."""

import pytest

from linak.integrations import orca_runner


def _fake_install(folder, names):
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"")
    return folder / "orca"


@pytest.mark.parametrize("modules", [
    ("orca", "orca_leanscf", "orca_scfgrad", "orca_startup"),  # ORCA 6.1: no orca_scf
    ("orca.exe", "orca_leanscf.exe", "orca_scfgrad.exe"),      # ORCA 6.1 on Windows
    ("orca", "orca_scf"),                                      # older releases
])
def test_orca_on_path_accepted(tmp_path, monkeypatch, modules):
    exe = _fake_install(tmp_path / "orca_dir", modules)
    monkeypatch.delenv("ORCA_EXE", raising=False)
    monkeypatch.setattr(orca_runner.shutil, "which", lambda _name: str(exe))
    assert orca_runner.resolve_orca_executable() == str(exe)


def test_screen_reader_on_path_rejected(tmp_path, monkeypatch):
    exe = _fake_install(tmp_path / "bin", ("orca", "ls"))
    monkeypatch.delenv("ORCA_EXE", raising=False)
    monkeypatch.setattr(orca_runner.shutil, "which", lambda _name: str(exe))
    with pytest.raises(RuntimeError, match="not the ORCA quantum-chemistry program"):
        orca_runner.resolve_orca_executable()
