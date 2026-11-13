"""The Delete species dialog and moving files to the Recycle Bin."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QFile
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from . import brand

TREE_LABELS = {
    "data_json": "JSON data",
    "plots": "Plots",
    "orca_outputs": "ORCA runs",
    "tables": "Imported and edited tables",
}


def _tree_size(paths: list[Path]) -> tuple[int, int]:
    """(file count, total bytes) of *paths*, walking into folders."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        else:
            files.append(path)
    return len(files), sum(p.stat().st_size for p in files)


def _size_text(bytes_total: int) -> str:
    """*bytes_total* in the largest unit that keeps it readable."""
    size = float(bytes_total)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def move_to_trash(paths: Iterable[Path]) -> list[Path]:
    """Move *paths* to the Recycle Bin; return those that could not be moved."""
    return [path for path in paths if not QFile.moveToTrash(str(path))]


def delete_permanently(paths: Iterable[Path]) -> list[Path]:
    """Delete *paths* for good; return those that could not be deleted."""
    failed = []
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError:
            failed.append(path)
    return failed


class DeleteSpeciesDialog(QDialog):
    """Asks whether to only unlist a species or also delete its data."""

    def __init__(self, species: str, files: dict[str, list[Path]],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Delete {species}")
        self.setWindowIcon(brand.species_icon(species))
        layout = QVBoxLayout(self)

        self.list_only = QRadioButton("Remove from the list only (keep all its files)")
        self.list_only.setChecked(True)
        layout.addWidget(self.list_only)

        self.delete_data = QRadioButton("Delete its data")
        layout.addWidget(self.delete_data)

        self.tree_boxes: dict[str, QCheckBox] = {}
        for tree, label in TREE_LABELS.items():
            count, size = _tree_size(files[tree])
            box = QCheckBox(f"{label} ({count} files, {_size_text(size)})")
            box.setChecked(count > 0)
            box.setEnabled(False)
            box.setProperty("has_files", count > 0)
            box.setStyleSheet("margin-left: 20px;")
            self.tree_boxes[tree] = box
            layout.addWidget(box)
        self.delete_data.setEnabled(any(files[tree] for tree in TREE_LABELS))
        self.delete_data.toggled.connect(self._on_mode_changed)

        note = QLabel(
            "Deleted files go to the Recycle Bin. Results shared between species, "
            "such as the comparison table, are not changed. To bring the species "
            f"back, type {species} in the species box: data shipped with Qali "
            "returns, and anything else can be restored from the Recycle Bin.")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Remove")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for box in self.tree_boxes.values():
            box.toggled.connect(self._update_ok_button)

    def showEvent(self, event) -> None:
        """Keep Enter on Cancel; Qt moves the default to Ok as the box is shown."""
        super().showEvent(event)
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        ok_button.setAutoDefault(False)
        ok_button.setDefault(False)
        self.buttons.button(QDialogButtonBox.Cancel).setDefault(True)

    def _on_mode_changed(self, deleting: bool) -> None:
        for box in self.tree_boxes.values():
            box.setEnabled(deleting and box.property("has_files"))
        self.buttons.button(QDialogButtonBox.Ok).setText("Delete" if deleting else "Remove")
        self._update_ok_button()

    def _update_ok_button(self) -> None:
        nothing_ticked = self.delete_data.isChecked() and not self.selected_trees()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not nothing_ticked)

    def selected_trees(self) -> tuple[str, ...]:
        """Trees to delete; empty when the species is only taken off the list."""
        if not self.delete_data.isChecked():
            return ()
        return tuple(tree for tree, box in self.tree_boxes.items()
                     if box.isEnabled() and box.isChecked())
