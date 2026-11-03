"""Console stream helpers for CLI pipelines.

LiNaK pipelines print Unicode arrows, Greek letters and spectroscopic terms.
On plain Windows consoles the default encoding is the legacy code page
(cp1252/cp936/...), which raises UnicodeEncodeError and kills the pipeline
halfway through output. Force UTF-8 on stdout/stderr when the stream allows
reconfiguration (real consoles) and leave redirected/piped handles alone.
"""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Best-effort switch of stdout/stderr to UTF-8 with lossy fallback."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
