"""Where Qali's files live on disk: bundled read-only data and the user home.

Bundled data (``resources/``) ships *inside* the ``qali`` package in every
build — the source checkout (``src/qali``), an installed wheel
(``site-packages/qali``) and the Nuitka bundle, whose spec places it at
``qali/resources`` in the payload. Nuitka keeps each compiled module's
``__file__`` inside that payload, so one package-relative rule resolves the
data everywhere; there is no frozen-mode branch.

Where a run's *outputs* go is :mod:`qali.core.workspace`'s job; this module
only provides the building blocks it uses.

Importing this module does nothing observable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


RESOURCES_DIR = "resources"
RUNTIME_DIR = ".qali"
# Set by the GUI for every stage subprocess (and usable from a shell): the
# folder a run writes into. Unset means the per-user data home.
PROJECT_ROOT_ENV = "QALI_PROJECT_ROOT"

# Qali was called LiNaK until 2026-11. Folders and variables made under the old
# name keep working: each is used when its new-name counterpart does not exist.
LEGACY_RUNTIME_DIR = ".linak"
LEGACY_PROJECT_ROOT_ENV = "LINAK_PROJECT_ROOT"
LEGACY_USER_DATA_DIR = "LiNaK"
USER_DATA_DIR = "Qali"


def project_root_from_env() -> str:
    """The folder named by ``QALI_PROJECT_ROOT`` (or the old ``LINAK_PROJECT_ROOT``); '' if unset."""
    return os.environ.get(PROJECT_ROOT_ENV) or os.environ.get(LEGACY_PROJECT_ROOT_ENV) or ""


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
    """Return whether *path* looks like a Qali source checkout."""
    return (path / "pyproject.toml").is_file() and (
        (path / "src/qali").is_dir() or (path / "src/linak").is_dir()
    )


def user_data_root(create: bool = True) -> Path:
    """Return Qali's per-user data home, creating it if needed (unless *create* is False)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    root = base / USER_DATA_DIR
    legacy = base / LEGACY_USER_DATA_DIR
    if not root.exists() and legacy.is_dir():
        return legacy                          # a data home made by LiNaK: keep using it
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_root_for(root: Path) -> Path:
    """Output root for a chosen project folder *root*.

    A checkout-shaped root works under ``<root>/.qali`` (or an existing
    ``<root>/.linak`` from before the rename) so pipeline writes
    never touch the source tree; any other folder receives outputs directly:
    ``<root>/data``, ``<root>/plots``, ``<root>/orca_outputs``.
    """
    if is_checkout(root):
        legacy = root / LEGACY_RUNTIME_DIR
        if legacy.is_dir() and not (root / RUNTIME_DIR).exists():
            return legacy                      # outputs made by LiNaK: keep using them
        return root / RUNTIME_DIR
    return root


def resource_root() -> Path:
    """The read-only ``resources/`` tree bundled inside the ``qali`` package."""
    return Path(__file__).resolve().parents[1] / RESOURCES_DIR


# Trees whose files the user may replace with an edited copy (Preferences > Tables and variables).
OVERRIDABLE_TREES = ("f_values", "NIST Levels", "Moments", "extrapolation")


def reference_path(*parts: str) -> Path:
    """Return a reference asset: the user's edited copy if there is one, else the bundled original."""
    if parts and parts[0] in OVERRIDABLE_TREES and not (len(parts) > 1 and parts[1] == "tables"):
        try:
            from qali.core import user_tables
            mine = user_tables.file_override(parts)
        except Exception:                      # an unreadable override never breaks a run
            mine = None
        if mine is not None:
            return mine
    return resource_root().joinpath(*parts)


def bundled_file(tree: str, folder: str | os.PathLike, name: str) -> str:
    """``folder/name``; when *folder* is the bundled *tree*, the user's edited copy of *name* wins."""
    try:
        if Path(folder).resolve() == (resource_root() / tree).resolve():
            return str(reference_path(tree, name))
    except OSError:
        pass
    return os.path.join(folder, name)
