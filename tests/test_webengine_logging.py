"""Regression pins for QtWebEngine's Chromium debug.log location."""

from __future__ import annotations

import os
import sys
import tempfile

from linak.gui.app import _configure_webengine_logging, webengine_log_path


def test_webengine_log_path_is_platform_temp(monkeypatch, tmp_path):
    """debug.log goes to the OS temp dir, never the working directory."""
    monkeypatch.setattr(tempfile, "tempdir", None)  # drop gettempdir()'s cache
    if sys.platform == "win32":
        fake = tmp_path / "fake-tmp"
        fake.mkdir()
        monkeypatch.setenv("TMP", str(fake))
        monkeypatch.setenv("TEMP", str(fake))
        assert str(webengine_log_path()) == str(fake / "LiNaK" / "debug.log")
    else:
        assert str(webengine_log_path()) == "/tmp/LiNaK/debug.log"


def test_configure_webengine_logging_sets_log_file(monkeypatch):
    # setenv (not delenv) so monkeypatch also undoes the value set below.
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "")
    _configure_webengine_logging()
    flags = os.environ["QTWEBENGINE_CHROMIUM_FLAGS"]
    assert "--log-file=" in flags
    assert flags.rsplit("=", 1)[1].endswith("debug.log")


def test_configure_webengine_logging_preserves_existing_flags(monkeypatch):
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")
    _configure_webengine_logging()
    flags = os.environ["QTWEBENGINE_CHROMIUM_FLAGS"]
    assert "--no-sandbox" in flags and "--log-file=" in flags


def test_configure_webengine_logging_honours_explicit_log_file(monkeypatch):
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--log-file=C:\\mine.log")
    _configure_webengine_logging()
    assert os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] == "--log-file=C:\\mine.log"
