"""Project discovery and output-file helpers for the LiNaK GUI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QMessageBox

from linak.core.paths import DATA_JSON_DIR, PLOTS_DIR, reference_path


def is_project_root(path: Path) -> bool:
    """Return whether *path* contains an installable LiNaK checkout."""
    return (path / "pyproject.toml").is_file() and (path / "src/linak").is_dir()


def rydberg_species(root: Path) -> list[str]:
    """Return species with a Rydberg dataset on disk."""
    species = []
    data_dir = root / DATA_JSON_DIR
    if data_dir.exists():
        suffix = "_rydberg.json"
        species.extend(p.name[: -len(suffix)] for p in data_dir.glob(f"*{suffix}"))
    
    # Also check bundled resources for reference species
    resources_data_dir = reference_path("data_json")
    if resources_data_dir.exists():
        suffix = "_rydberg.json"
        species.extend(p.name[: -len(suffix)] for p in resources_data_dir.glob(f"*{suffix}"))
    
    return sorted(set(species))


def species_outputs(root: Path, species: str) -> list[tuple[Path, str]]:
    """Return plot and JSON output files for one species."""
    items: list[tuple[Path, str]] = []
    plot_dir = root / PLOTS_DIR / species
    if plot_dir.exists():
        items.extend((path, "PLOT") for path in sorted(plot_dir.glob("*.html")))

    data_dir = root / DATA_JSON_DIR
    if data_dir.exists():
        items.extend(
            (path, "JSON")
            for path in sorted(data_dir.glob(f"{species}_*.json"))
        )
        exact = data_dir / f"{species}.json"
        if exact.exists():
            items.append((exact, "JSON"))
    return items


def shared_outputs(root: Path) -> list[tuple[Path, str]]:
    """Return cross-element output files from the project root."""
    items: list[tuple[Path, str]] = []
    plots_root = root / PLOTS_DIR
    if plots_root.exists():
        items.extend((path, "PLOT") for path in sorted(plots_root.glob("*.html")))

    data_dir = root / DATA_JSON_DIR
    if data_dir.exists():
        for name in ("comparison_table.json", "nuclear_data.json"):
            path = data_dir / name
            if path.exists():
                items.append((path, "JSON"))
    return items


def open_pipeline_overview_html(root: Path, parent_widget) -> None:
    """Open the project's Pipeline Overview document in the system browser."""
    htmls_dir = reference_path("HTMLs")
    candidates = (
        list(htmls_dir.glob("*ipeline*verview*.html"))
        if htmls_dir.exists()
        else []
    )
    if not candidates:
        QMessageBox.information(
            parent_widget,
            "Not found",
            f"Couldn't find a Pipeline Overview HTML file under {htmls_dir}. "
            "Looking for a name like 'Pipeline Overview.html'.",
        )
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(candidates[0])))


def is_within_project(root: Path, candidate: Path) -> bool:
    """Return whether *candidate* resolves inside *root*."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve(strict=False)
    except OSError:
        return False
    return (
        candidate_resolved == root_resolved
        or root_resolved in candidate_resolved.parents
    )


def ensure_project_safe_path(root: Path, target: Path) -> Path:
    """Validate a write target and return it as an expanded path."""
    target = target.expanduser()
    if not is_within_project(root, target):
        raise ValueError(
            f"Refusing to write outside the LiNaK project folder:\n{target}\n\n"
            f"Allowed root: {root}"
        )
    return target


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically beside the destination file."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
