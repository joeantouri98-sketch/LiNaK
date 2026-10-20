"""Application entry point for the LiNaK desktop GUI."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def webengine_log_path() -> Path:
    """Return the OS-specific path of QtWebEngine's Chromium debug.log.

    ``%TEMP%\\LiNaK\\debug.log`` on Windows, ``/tmp/LiNaK/debug.log`` on
    macOS and Linux, so the log never lands in the working directory.
    """
    if sys.platform == "win32":
        base = Path(tempfile.gettempdir())
    else:
        base = Path("/tmp")
    return base / "LiNaK" / "debug.log"


def _configure_webengine_logging() -> None:
    """Send Chromium's debug.log to the OS temp dir instead of the CWD.

    QtWebEngine defaults to writing ``debug.log`` next to the executable /
    working directory; point ``--log-file`` at the temp dir before any
    WebEngine context exists. An existing ``QTWEBENGINE_CHROMIUM_FLAGS``
    value is preserved and only extended if it does not set a log file.
    """
    existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--log-file" in existing:
        return
    log_dir = webengine_log_path().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    flag = f"--log-file={log_dir / 'debug.log'}"
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{existing} {flag}".strip()


def main() -> int:
    _configure_webengine_logging()
    from .main_window import main as run_window

    return run_window()


if __name__ == "__main__":
    sys.exit(main())
