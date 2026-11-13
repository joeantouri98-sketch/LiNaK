"""Atomic, validated file writes shared by the pipelines and the GUI.

A write lands in a temporary file beside its target and is renamed into
place, so a reader (the next pipeline stage, the GUI) never sees a
half-written file. No newline translation happens, so outputs are
byte-identical on every OS. JSON follows the repository rules: UTF-8 and
``allow_nan=False``, so a NaN/Infinity is a hard error instead of an invalid
document.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, "os.PathLike[str]"]

# One path component built from user input (run tags, output suffixes): no
# separators, drive letters ("C:x") or NTFS stream syntax ("a:b").
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9._+-]+")


def is_safe_name(name: str) -> bool:
    """True if *name* can be used as a single file/folder name anywhere."""
    return bool(_SAFE_NAME_RE.fullmatch(name)) and name not in (".", "..")


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically, creating parent directories."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=path.suffix, dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_json(path: PathLike, payload: Any, *, indent: int | None = 2, **dumps_kwargs: Any) -> None:
    """Serialise *payload* with ``allow_nan=False`` and write it atomically."""
    atomic_write_text(path, json.dumps(payload, indent=indent, allow_nan=False, **dumps_kwargs))
