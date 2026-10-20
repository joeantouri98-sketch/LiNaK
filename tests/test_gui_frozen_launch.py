# -*- coding: utf-8 -*-
"""The frozen GUI must relaunch the *bundle* exe for pipeline stages.

Regression context: in a Nuitka onefile build ``sys.frozen`` may be falsy
and ``sys.executable`` may quote the unpacked payload inside the per-run
temp directory, which does not exist on disk — stage launches then died
with "The system cannot find the file specified".
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from linak.core.paths import is_frozen_app
from linak.gui.stage_process import cli_log_paths, frozen_executable


def test_frozen_executable_prefers_existing_argv0(tmp_path, monkeypatch):
    exe = tmp_path / "LiNaK.exe"
    exe.write_bytes(b"MZ")  # the name/suffix is what matters, not content
    monkeypatch.setattr(sys, "argv", [str(exe), "--pipeline", "feshbach"], raising=False)
    monkeypatch.setattr(
        sys, "executable", str(tmp_path / "payload" / "python.exe"), raising=False
    )
    assert frozen_executable() == str(exe.resolve())


def test_frozen_executable_skips_missing_paths(tmp_path, monkeypatch):
    """sys.executable quoting a nonexistent payload must not be 'fixed up'.

    With neither argv[0] nor sys.executable usable and no Nuitka onefile
    parent recorded, the last-resort answer is sys.executable unchanged.
    """
    missing = tmp_path / "payload" / "python.exe"
    monkeypatch.setattr(sys, "argv", ["__main__.py"], raising=False)
    monkeypatch.setattr(sys, "executable", str(missing), raising=False)
    monkeypatch.delenv("NUITKA_ONEFILE_PARENT", raising=False)
    assert not missing.is_file()
    assert frozen_executable() == str(missing)


def test_cli_log_paths_match_the_bundled_spec(tmp_path, monkeypatch):
    """The GUI tail and the bundle's forced-stdout spec must agree."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    stdout_log, stderr_log = cli_log_paths(1234)
    assert stdout_log.name == "LiNaK_stdout_1234.log"
    assert stderr_log.name == "LiNaK_stderr_1234.log"
    assert stdout_log.parent == Path(tempfile.gettempdir())


def test_is_frozen_app_probes_nuitka_compiled_main(monkeypatch):
    import types

    monkeypatch.setattr(sys, "frozen", False, raising=False)
    fake_main = types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert is_frozen_app() is False
    fake_main.__compiled__ = "Nuitka"  # what Nuitka injects
    assert is_frozen_app() is True


def test_frozen_executable_accepts_a_suffixless_bundle(tmp_path, monkeypatch):
    """macOS/Linux bundles have no ``.exe`` suffix; they must still relaunch."""
    exe = tmp_path / "LiNaK.bin"
    exe.write_bytes(b"\x7fELF")
    monkeypatch.setattr(sys, "argv", [str(exe)], raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "payload" / "python"), raising=False)
    assert frozen_executable() == str(exe.resolve())
