"""Species > Properties: the Tables and variables page restricted to one element."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QPushButton, QVBoxLayout

from .new_species_dialog import NewSpeciesDialog
from .tables_page import TablesPage


def element_of_species(species: str) -> str:
    """``Na_c1`` -> ``Na``: the element symbol a species id starts with."""
    return species.split("_")[0]


class PropertiesDialog(QDialog):
    def __init__(self, root: Path, species: str, parent=None):
        super().__init__(parent)
        self.species = species
        self.element = element_of_species(species)
        self.root = Path(root)
        self.setWindowTitle(f"Properties of {species}")
        self.resize(980, 600)
        self.lay = QVBoxLayout(self)
        self.page = TablesPage(root, element=self.element, species=species)
        self.lay.addWidget(self.page, 1)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.buttons.accepted.connect(self._ok)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.page.reset_all)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).setToolTip("Restore the table or file you are on.")
        self.setup_btn = QPushButton("Set up missing data...")
        self.setup_btn.setToolTip("Import the NIST files and enter the numbers this species lacks.")
        self.setup_btn.clicked.connect(self._setup)
        self.buttons.addButton(self.setup_btn, QDialogButtonBox.ActionRole)
        self.lay.addWidget(self.buttons)

    def _setup(self) -> None:
        if not self._apply():
            return
        NewSpeciesDialog(self.root, self.species, self).exec()
        self.lay.removeWidget(self.page)                       # the page loaded its state before the set-up changed it
        self.page.deleteLater()
        self.page = TablesPage(self.root, element=self.element, species=self.species)
        self.lay.insertWidget(0, self.page, 1)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.disconnect()
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.page.reset_all)

    def _apply(self) -> bool:
        problems = self.page.apply()
        if problems:
            QMessageBox.warning(self, "Properties", "\n".join(problems))
        return not problems

    def _ok(self) -> None:
        if self._apply():
            self.accept()
