"""Run a stage under a name: the outputs it writes are kept as ``<file>__<name>.<ext>`` and the usual files stay as they were.

Lets the same stage be run with different settings (a hot and a cold blackbody run, two Grotrian overlays) without one
run overwriting the other. The stage itself is unchanged: it writes its usual files, and when the run ends the changed
ones are renamed and the earlier versions put back. Stages that read another stage's output keep reading the usual
file, so a named run never changes what the rest of the pipeline sees.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

SEPARATOR = "__"
MAX_LEN = 40
_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(name: str) -> str:
    """A file-name-safe version of *name*; empty when nothing usable is left. Spaces become underscores."""
    cleaned = _BAD.sub("_", (name or "").strip()).strip("._-")
    return cleaned[:MAX_LEN].strip("._-")


def named_path(path: Path, name: str) -> Path:
    """``data/Na_blackbody.json`` + ``hot`` -> ``data/Na_blackbody__hot.json``."""
    path = Path(path)
    return path.with_name(f"{path.stem}{SEPARATOR}{name}{path.suffix}")


def _signature(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


class OutputSnapshot:
    """The output files of a stage before a run, so a run can be filed under a name afterwards."""

    def __init__(self, paths, backup_root: Path | None = None):
        self.paths = [Path(p) for p in dict.fromkeys(paths)]
        self._before = {p: _signature(p) for p in self.paths}
        self._backup = Path(tempfile.mkdtemp(prefix="qali_saveas_", dir=backup_root))
        self._saved: dict[Path, Path] = {}
        for i, p in enumerate(self.paths):
            if self._before[p] is not None:
                copy = self._backup / f"{i}_{p.name}"
                shutil.copy2(p, copy)
                self._saved[p] = copy

    def changed(self) -> list[Path]:
        """Files the run wrote or rewrote."""
        return [p for p in self.paths if _signature(p) is not None and _signature(p) != self._before[p]]

    def finish(self, name: str | None, ok: bool = True) -> list[Path]:
        """Put the usual files back; with *name* and a successful run, keep the new ones as named files.

        Returns the named files written. A failed run keeps nothing.
        """
        kept: list[Path] = []
        try:
            for p in self.changed():
                if ok and name:
                    target = named_path(p, name)
                    os.replace(p, target)
                    kept.append(target)
            for p in self.paths:
                if p in self._saved:
                    if _signature(p) != self._before[p]:
                        shutil.copy2(self._saved[p], p)
                elif p.exists() and self._before[p] is None:
                    p.unlink()
        finally:
            shutil.rmtree(self._backup, ignore_errors=True)
        return kept
