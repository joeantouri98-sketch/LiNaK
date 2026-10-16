"""Paths used by the packaged application and development checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


RESOURCES_DIR = "resources"
DATA_JSON_DIR = f"{RESOURCES_DIR}/data_json"
PLOTS_DIR = f"{RESOURCES_DIR}/plots"
BASIS_SETS_DIR = f"{RESOURCES_DIR}/Basis sets"
ORCA_OUTPUTS_DIR = f"{RESOURCES_DIR}/orca_outputs"


def project_root() -> Path:
    """Return the selected project root for legacy pipeline files."""
    configured = os.environ.get("LINAK_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def resource_root() -> Path:
    """Return the read-only resources bundled with the application."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[3]


def resource_path(*parts: str) -> Path:
    root = resource_root()
    path = root.joinpath(*parts)
    if path.exists() or not parts or parts[0] != "resources":
        return path
    return root.joinpath(*parts[1:])


def reference_path(*parts: str) -> Path:
    """Return a read-only reference asset from the resources directory."""
    return resource_path(RESOURCES_DIR, *parts)


def user_data_root() -> Path:
    """Return a writable per-user directory for generated application data."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    root = base / "LiNaK"
    root.mkdir(parents=True, exist_ok=True)
    return root
