"""Project discovery and output-file helpers for the LiNaK GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QMessageBox

from linak.core.paths import reference_path
from linak.core.species import is_ion_id
from linak.core.workspace import Workspace


def rydberg_species(root: Path) -> list[str]:
    """Return the species with a Rydberg dataset in the folder *root*."""
    suffix = "_rydberg.json"
    data_dir = Workspace(root=root).data_json_dir
    return sorted(p.name[: -len(suffix)] for p in data_dir.glob(f"*{suffix}"))


def _belongs_to_ion_of(species: str, stem: str) -> bool:
    """True for ``Na_c1_lifetimes`` under ``Na``: the ion's file, not the atom's."""
    first = stem[len(species) + 1:].split("_", 1)[0]
    return is_ion_id(f"{species}_{first}")


def species_outputs(root: Path, species: str) -> list[tuple[Path, str]]:
    """Return plot and JSON output files for one species."""
    items: list[tuple[Path, str]] = []
    ws = Workspace(root=root)
    plot_dir = ws.plots_dir / species
    if plot_dir.exists():
        items.extend((path, "PLOT") for path in sorted(plot_dir.glob("*.html")))

    data_dir = ws.data_json_dir
    if data_dir.exists():
        items.extend(
            (path, "JSON")
            for path in sorted(data_dir.glob(f"{species}_*.json"))
            if not _belongs_to_ion_of(species, path.stem)
        )
        exact = data_dir / f"{species}.json"
        if exact.exists():
            items.append((exact, "JSON"))
    return items


def shared_outputs(root: Path) -> list[tuple[Path, str]]:
    """Return cross-element output files from the project root."""
    items: list[tuple[Path, str]] = []
    ws = Workspace(root=root)
    plots_root = ws.plots_dir
    if plots_root.exists():
        items.extend((path, "PLOT") for path in sorted(plots_root.glob("*.html")))

    data_dir = ws.data_json_dir
    if data_dir.exists():
        for name in ("comparison_table.json", "nuclear_data.json"):
            path = data_dir / name
            if path.exists():
                items.append((path, "JSON"))
    return items


def open_pipeline_overview_html(root: Path, parent_widget) -> None:
    """Open the project's Pipeline Overview document in the system browser.

    Looks under the selected project root first, then falls back to the
    bundled reference copy when the project has no ``HTMLs/`` directory.
    """
    search_dirs = (root / "HTMLs", reference_path("HTMLs"))
    candidates = [
        path
        for htmls_dir in search_dirs
        if htmls_dir.exists()
        for path in sorted(htmls_dir.glob("*ipeline*verview*.html"))
    ]
    if not candidates:
        looked_in = ", ".join(str(d) for d in search_dirs)
        QMessageBox.information(
            parent_widget,
            "Not found",
            f"Couldn't find a Pipeline Overview HTML file under {looked_in}. "
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
            f"Refusing to write outside the output folder:\n{target}\n\n"
            f"Allowed root: {root}"
        )
    return target
