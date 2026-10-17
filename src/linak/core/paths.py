"""Paths used by the packaged application and development checkout."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


RESOURCES_DIR = "resources"
RUNTIME_DIR = ".linak"


def _is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src/linak").is_dir()


def user_data_root() -> Path:
    """Return a writable per-user directory for application data."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    root = base / "LiNaK"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_root() -> Path:
    """Return the selected project root, or the current directory."""
    configured = os.environ.get("LINAK_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd()


def writable_root() -> Path:
    """Return the project-local or per-user writable data directory."""
    root = project_root()
    if _is_checkout(root):
        return root / RUNTIME_DIR
    return user_data_root()


def resource_root() -> Path:
    """Return the read-only resources bundled with the application."""
    if getattr(sys, "frozen", False):
        # PyInstaller sets _MEIPASS
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        
        # Nuitka macOS app bundle: executable is in .../Contents/MacOS/
        # Resources are in .../Contents/Resources/
        executable = Path(sys.executable)
        if executable.parent.name == "MacOS":
            bundle_root = executable.parents[1]  # .../Contents/
            resources_dir = bundle_root / "Resources"
            if resources_dir.is_dir():
                return resources_dir
        
        # Default: executable's parent directory
        return executable.parent

    checkout_root = Path(__file__).resolve().parents[3]
    if (checkout_root / RESOURCES_DIR).is_dir():
        return checkout_root

    installed_root = Path(sys.prefix)
    if (installed_root / RESOURCES_DIR).is_dir():
        return installed_root

    return checkout_root


def resource_path(*parts: str) -> Path:
    root = resource_root()
    path = root.joinpath(*parts)
    if path.exists() or not parts or parts[0] != "resources":
        return path
    return root.joinpath(*parts[1:])


def reference_path(*parts: str) -> Path:
    """Return a read-only reference asset from the resources directory."""
    return resource_path(RESOURCES_DIR, *parts)


DATA_JSON_DIR = str(writable_root() / "data")
PLOTS_DIR = str(writable_root() / "plots")
BASIS_SETS_DIR = str(reference_path("Basis sets"))
ORCA_OUTPUTS_DIR = str(writable_root() / "orca_outputs")


def _seed_data_dir() -> None:
    """Copy bundled/reference JSON inputs into the writable data directory."""
    source = reference_path("data_json")
    target = Path(DATA_JSON_DIR)
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


_seed_data_dir()
