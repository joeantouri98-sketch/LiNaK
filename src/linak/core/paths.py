"""Where LiNaK's files live on disk: bundled read-only data and the user home.

Bundled data (``resources/``) ships *inside* the ``linak`` package in every
build — the source checkout (``src/linak``), an installed wheel
(``site-packages/linak``) and the Nuitka bundle, whose spec places it at
``linak/resources`` in the payload. Nuitka keeps each compiled module's
``__file__`` inside that payload, so one package-relative rule resolves the
data everywhere; there is no frozen-mode branch.

Where a run's *outputs* go is :mod:`linak.core.workspace`'s job; this module
only provides the building blocks it uses.

Importing this module does nothing observable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


RESOURCES_DIR = "resources"
RUNTIME_DIR = ".linak"
# Set by the GUI for every stage subprocess (and usable from a shell): the
# folder a run writes into. Unset means the per-user data home.
PROJECT_ROOT_ENV = "LINAK_PROJECT_ROOT"


def is_frozen_app() -> bool:
    """True when running from the compiled bundle rather than a Python env.

    Nuitka does not set ``sys.frozen``; it injects ``__compiled__`` into the
    ``__main__`` module instead, so probe both.
    """
    main_module = sys.modules.get("__main__")
    return bool(
        getattr(sys, "frozen", False)
        or (main_module is not None and hasattr(main_module, "__compiled__"))
    )


def is_checkout(path: Path) -> bool:
    """Return whether *path* looks like a LiNaK source checkout."""
    return (path / "pyproject.toml").is_file() and (path / "src/linak").is_dir()


def user_data_root() -> Path:
    """Return LiNaK's per-user data home, creating it if needed."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    root = base / "LiNaK"
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_root_for(root: Path) -> Path:
    """Output root for a chosen project folder *root*.

    A checkout-shaped root works under ``<root>/.linak`` so pipeline writes
    never touch the source tree; any other folder receives outputs directly:
    ``<root>/data``, ``<root>/plots``, ``<root>/orca_outputs``.
    """
    if is_checkout(root):
        return root / RUNTIME_DIR
    return root


def resource_root() -> Path:
    """The read-only ``resources/`` tree bundled inside the ``linak`` package."""
    return Path(__file__).resolve().parents[1] / RESOURCES_DIR


def reference_path(*parts: str) -> Path:
    """Return a read-only reference asset from the bundled resources."""
    return resource_root().joinpath(*parts)
