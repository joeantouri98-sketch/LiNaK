"""Application entry point for the LiNaK desktop GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_frozen_qt_webengine() -> None:
    """Point QtWebEngine at its helper inside a frozen app bundle."""
    if not getattr(sys, "frozen", False) or os.environ.get("QTWEBENGINEPROCESS_PATH"):
        return

    executable = Path(sys.executable).resolve()
    candidates = [
        executable.parents[1]
        / "Frameworks/QtWebEngineCore.framework/Versions/A/Helpers"
        / "QtWebEngineProcess.app/Contents/MacOS/QtWebEngineProcess",
        executable.parent / "PySide6/Qt/lib/QtWebEngineCore.framework/Helpers"
        / "QtWebEngineProcess.app/Contents/MacOS/QtWebEngineProcess",
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["QTWEBENGINEPROCESS_PATH"] = str(candidate)
            return


def _configure_frozen_qt_webengine_resources() -> None:
    if not getattr(sys, "frozen", False) or os.environ.get("QTWEBENGINE_RESOURCES_PATH"):
        return

    executable = Path(sys.executable).resolve()
    candidates = [
        executable.parents[1]
        / "Frameworks/QtWebEngineCore.framework/Versions/A/Resources",
        executable.parent / "PySide6/Qt/lib/QtWebEngineCore.framework/Resources",
    ]
    for candidate in candidates:
        if (candidate / "icudtl.dat").exists():
            os.environ["QTWEBENGINE_RESOURCES_PATH"] = str(candidate)
            locales = candidate / "qtwebengine_locales"
            if locales.is_dir():
                os.environ["QTWEBENGINE_LOCALES_PATH"] = str(locales)
            return


def main() -> int:
    _configure_frozen_qt_webengine()
    _configure_frozen_qt_webengine_resources()
    from .main_window import main as run_window

    return run_window()


if __name__ == "__main__":
    sys.exit(main())
